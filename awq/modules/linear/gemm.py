import torch
import warnings
import torch.nn as nn
from torch.autograd import Function
from awq.utils.module import try_import
from awq.utils.packing_utils import dequantize_gemm

# NOTE: We check if awq_ext or triton is available. awq_ext will be preferred if both are installed.

awq_ext, msg = try_import("awq_ext")
user_has_been_warned = False

try:
    from awq.modules.triton.gemm import awq_gemm_triton, awq_dequantize_triton

    # covers CUDA, ROCm and XPU. If we can import triton, then we can use it.
    TRITON_AVAILABLE = True

except ImportError:
    TRITON_AVAILABLE = False

# Adapted from https://github.com/compressa-ai/AutoAWQ/tree/dev
class WQLinearMMFunction(Function):
    @staticmethod
    # ctx is the first argument to forward
    def forward(
        ctx,
        x,
        qweight,
        qzeros,
        scales,
        w_bit=4,
        group_size=128,
        bias=None,
        out_features=0,
    ):
        # The forward pass can use ctx.
        ctx.save_for_backward(x, qweight, qzeros, scales, bias)
        ctx.out_features = out_features

        out_shape = x.shape[:-1] + (out_features,)
        x = x.to(torch.float16)

        if awq_ext is not None:
            FP16_MATMUL_HEURISTIC_CONDITION = x.shape[0] * x.shape[1] >= 1024

            if FP16_MATMUL_HEURISTIC_CONDITION:
                out = awq_ext.dequantize_weights_cuda(
                    qweight, scales, qzeros, 0, 0, 0, False
                )
                out = torch.matmul(x, out)
            else:
                out = awq_ext.gemm_forward_cuda(
                    x.reshape(-1, x.shape[-1]), qweight, scales, qzeros, 8
                )

        elif TRITON_AVAILABLE:
            FP16_MATMUL_HEURISTIC_CONDITION = x.shape[0] * x.shape[1] >= 1024

            if FP16_MATMUL_HEURISTIC_CONDITION:
                out = awq_dequantize_triton(qweight, scales, qzeros)
                out = torch.matmul(x, out)
            else:
                out = awq_gemm_triton(
                    x.reshape(-1, x.shape[-1]), qweight, scales, qzeros, split_k_iters=8,
                )

        else:
            if not user_has_been_warned:
                warnings.warn("Using naive (slow) implementation." + msg)
                user_has_been_warned = True
            out = dequantize_gemm(qweight, qzeros, scales, w_bit, group_size)
            out = torch.matmul(x, out)

        out = out + bias if bias is not None else out
        out = out.reshape(out_shape)

        # always want 3D tensor if tensor is 2D
        if len(out.shape) == 2:
            out = out.unsqueeze(0)

        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, qweight, qzeros, scales, bias = ctx.saved_tensors

        if awq_ext is None and not TRITON_AVAILABLE:
            raise ValueError(
                "either triton or autoawq-kernels is needed to be installed to use `.backward()`. Make sure to install the auto-awq kernels"
                " by following the installation guides in https://github.com/casper-hansen/AutoAWQ_kernels"
            )
        
        # Cast to correct dtype for mixed precision training
        if awq_ext is not None:
            weights = awq_ext.dequantize_weights_cuda(
                qweight, scales, qzeros, 1, 0, 0, False
            ).to(grad_output.dtype)
        else:
            weights = awq_dequantize_triton(
                qweight, scales, qzeros
            ).to(grad_output.dtype)

        if ctx.needs_input_grad[0]:
            # 3D matmul using torch.bmm: https://pytorch.org/docs/stable/generated/torch.bmm.html#torch.bmm
            # to propagate gradient across all batch sizes.
            batch_size = grad_output.shape[0]
            grad_input = grad_output.bmm(weights.transpose(0, 1).unsqueeze(0).repeat(batch_size, 1, 1))

        return grad_input, None, None, None, None, None, None, None

class WQLinear_GEMM(nn.Module):
    def __init__(
        self, w_bit, group_size, in_features, out_features, bias, dev, training=False
    ):
        super().__init__()

        if w_bit not in [4]:
            raise NotImplementedError("Only 4-bit are supported for now.")

        self.in_features = in_features
        self.out_features = out_features
        self.w_bit = w_bit
        self.group_size = group_size if group_size != -1 else in_features
        self.training = training

        # quick sanity check (make sure aligment)
        assert self.in_features % self.group_size == 0
        assert out_features % (32 // self.w_bit) == 0

        self.register_buffer(
            "qweight",
            torch.zeros(
                (in_features, out_features // (32 // self.w_bit)),
                dtype=torch.int32,
                device=dev,
            ),
        )
        self.register_buffer(
            "qzeros",
            torch.zeros(
                (in_features // self.group_size, out_features // (32 // self.w_bit)),
                dtype=torch.int32,
                device=dev,
            ),
        )
        self.register_buffer(
            "scales",
            torch.zeros(
                (in_features // self.group_size, out_features),
                dtype=torch.float16,
                device=dev,
            ),
        )
        if bias:
            self.register_buffer(
                "bias",
                torch.zeros(
                    (out_features),
                    dtype=torch.float16,
                    device=dev,
                ),
            )
        else:
            self.bias = None

    @classmethod
    def from_linear(
        cls, linear, w_bit, group_size, init_only=False, scales=None, zeros=None
    ):
        awq_linear = cls(
            w_bit,
            group_size,
            linear.in_features,
            linear.out_features,
            linear.bias is not None,
            linear.weight.device,
        )
        if init_only:  # just prepare for loading sd
            return awq_linear

        # need scales and zeros info for real quantization
        assert scales is not None and zeros is not None
        scale_zeros = zeros * scales

        awq_linear.scales = scales.clone().half()
        if linear.bias is not None:
            awq_linear.bias = linear.bias.clone().half()

        pack_num = 32 // awq_linear.w_bit

        group_indices = torch.arange(awq_linear.in_features) // group_size
        intweight = torch.round(
            (linear.weight.data.t() + scale_zeros[group_indices]) / awq_linear.scales[group_indices]
        ).to(torch.int32).contiguous()  # shape: [in_features, out_features]

        if awq_linear.w_bit == 4:
            order_map = [0, 2, 4, 6, 1, 3, 5, 7]
        else:
            raise NotImplementedError("Only 4-bit are supported for now.")

        # [0, 4, 8, 12, 16, 20, 24, 28]
        shifts = torch.arange(0, pack_num * awq_linear.w_bit, awq_linear.w_bit, device=intweight.device)
        # 重塑 -> 重排 -> 位移 -> 合并
        qweight = (intweight.view(intweight.shape[0], -1, pack_num)[:, :, order_map] << shifts.view(1, 1, -1)).sum(dim=2).to(torch.int32)

        awq_linear.qweight = qweight

        zeros = zeros.to(dtype=torch.int32)

        if awq_linear.w_bit == 4:
            order_map = [0, 2, 4, 6, 1, 3, 5, 7]
        else:
            raise NotImplementedError("Only 4-bit are supported for now.")

        # [0, 4, 8, 12, 16, 20, 24, 28]
        shifts = torch.arange(0, pack_num * awq_linear.w_bit, awq_linear.w_bit, device=zeros.device)
        # 重塑 -> 重排 -> 位移 -> 合并
        qzeros = (zeros.view(zeros.shape[0], -1, pack_num)[:, :, order_map] << shifts.view(1, 1, -1)).sum(dim=2).to(torch.int32)

        awq_linear.qzeros = qzeros

        return awq_linear

    def forward(self, x):
        out_shape = x.shape[:-1] + (self.out_features,)

        input_dtype = x.dtype
        if input_dtype != torch.float16:
            x = x.half()

        if self.training:
            out = WQLinearMMFunction.apply(
                x,
                self.qweight,
                self.qzeros,
                self.scales,
                self.w_bit,
                self.group_size,
                self.bias,
                self.out_features,
            )
        else:
            with torch.no_grad():
                out = WQLinearMMFunction.apply(
                    x,
                    self.qweight,
                    self.qzeros,
                    self.scales,
                    self.w_bit,
                    self.group_size,
                    self.bias,
                    self.out_features,
                )

        if input_dtype != torch.float16:
            out = out.to(dtype=input_dtype)

        return out.reshape(out_shape)

    def extra_repr(self) -> str:
        return (
            "in_features={}, out_features={}, bias={}, w_bit={}, group_size={}".format(
                self.in_features,
                self.out_features,
                self.bias is not None,
                self.w_bit,
                self.group_size,
            )
        )
