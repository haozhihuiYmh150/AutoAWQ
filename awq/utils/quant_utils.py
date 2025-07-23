import torch
import contextlib
import logging
import torch.nn as nn
from typing import Dict, List, Optional
from awq.modules.linear import (
    WQLinear_GEMM,
    WQLinear_GEMV,
    WQLinear_Marlin,
    WQLinear_GEMVFast,
)
from awq.utils.module import (
    set_op_by_name,
)
from awq.utils.kernels import (
    weight_dequant,
)
logging.basicConfig(
    level=logging.WARNING,
    format='%(lineno)d: %(message)s',
)

Q_BITS = 4
STORAGE_BITS = 32
PACK_NUM = STORAGE_BITS // Q_BITS

ORDINAL_PACK_ORDER = [0, 1, 2, 3, 4, 5, 6, 7]
AWQ_PACK_ORDER = [0, 2, 4, 6, 1, 3, 5, 7]
REVERSE_AWQ_PACK_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]


def pack(imatrix: torch.Tensor, direction: str = "column"):
    """
    Packs a 4-bit integer matrix into a packed 32-bit integer matrix.
    Args:
        imatrix (torch.Tensor): matrix of integers
        direction (str): direction of packing, either "column" or "row"

    Returns:
        qmatrix (torch.Tensor): packed matrix of integers
    """
    shifts = torch.arange(0, STORAGE_BITS, Q_BITS, device=imatrix.device)

    imatrix = imatrix.to(torch.int8)
    imatrix = torch.bitwise_and(imatrix, 0x0F)  # eventually correct overflow

    if direction == "column":
        imatrix = imatrix.view(-1, imatrix.shape[1] // PACK_NUM, PACK_NUM)
        qmatrix = torch.bitwise_left_shift(imatrix, shifts[None, None, :]).sum(dim=-1)

    elif direction == "row":
        imatrix = imatrix.view(imatrix.shape[0] // PACK_NUM, PACK_NUM, -1)
        qmatrix = torch.bitwise_left_shift(imatrix, shifts[None, :, None]).sum(dim=1)

    qmatrix = qmatrix.to(torch.int32)

    return qmatrix


def unpack(qmatrix: torch.Tensor, direction: str = "column"):
    """
    Unpacks a 32-bit packed integer matrix into a 4-bit integer matrix.

    Args:
        qmatrix (torch.Tensor): matrix of packed integers
        direction (str): direction of unpacking, either "column" or "row"

    Returns:
        imatrix (torch.Tensor): matrix of integers
    """
    shifts = torch.arange(0, STORAGE_BITS, Q_BITS, device=qmatrix.device)

    if direction == "column":
        imatrix = torch.bitwise_right_shift(
            qmatrix[:, :, None], shifts[None, None, :]
        ).view(qmatrix.shape[0], -1)

    elif direction == "row":
        imatrix = torch.bitwise_right_shift(
            qmatrix[:, None, :], shifts[None, :, None]
        ).view(-1, qmatrix.shape[-1])

    imatrix = imatrix.to(torch.int8) & 0x0F  # eventually correct overflow

    return imatrix


def quantize(fmatrix, scales, zeros, group_size):
    """
    Quantizes a matrix of 16-bit floats into a matrix of 4-bit integers.

    Args:
        fmatrix (torch.Tensor): matrix of 16-bit floats
        scales (torch.Tensor): matrix of 16-bit floats
        zeros (torch.Tensor): matrix of 4-bit integers
        group_size (int): group size

    Returns:
        imatrix (torch.Tensor): matrix of 4-bit integers
    """
    zeros = zeros.to(torch.int8) & 0x0F

    imatrix = torch.round(
        (
            fmatrix / scales.repeat_interleave(group_size, dim=0)
            + zeros.repeat_interleave(group_size, dim=0)
        )
    )

    imatrix = imatrix.to(torch.int8) & 0x0F

    return imatrix


def dequantize(imatrix, scales, zeros, group_size):
    """
    Dequantizes a 4-bit integer matrix into a float matrix.

    Args:
        imatrix (torch.Tensor): matrix of 4-bit integers
        scales (torch.Tensor): matrix of 16-bit floats
        zeros (torch.Tensor): matrix of 4-bit integers
        group_size (int): group size

    Returns:
        fmatrix (torch.Tensor): matrix of 16-bit floats
    """
    zeros = zeros.to(torch.int8) & 0x0F
    imatrix = imatrix.to(torch.int8) & 0x0F

    fmatrix = (
        imatrix - zeros.repeat_interleave(group_size, dim=0)
    ) * scales.repeat_interleave(group_size, dim=0)

    fmatrix = fmatrix.to(torch.float16)

    return fmatrix


def apply_order(
    imatrix: torch.Tensor,
    direction: str = "column",
    order: List[int] = ORDINAL_PACK_ORDER,
):
    """
    Applies the order to a 4-bit integer matrix.

    Args:
        imatrix (torch.Tensor): matrix of integers
        direction (str): direction of applying order, either "column" or "row"
        order (List[int]): order to apply, default is ordinal packing order

    Returns:
        imatrix (torch.Tensor): matrix of integers
    """
    if direction == "column":
        imatrix = imatrix.view(-1, PACK_NUM)[:, order].view(imatrix.shape)
    elif direction == "row":
        imatrix = imatrix.view(PACK_NUM, -1)[order, :].view(imatrix.shape)

    return imatrix


def awq_to_exllama(qweight, qzeros):
    # awq uses column packing for both weights and zeros
    izeros = unpack(qzeros, direction="column")
    iweights = unpack(qweight, direction="column")

    # Reverse the order of the iweight and izeros tensors
    izeros = apply_order(izeros, direction="column", order=REVERSE_AWQ_PACK_ORDER)
    iweights = apply_order(iweights, direction="column", order=REVERSE_AWQ_PACK_ORDER)
    # Subtract 1 from the izeros tensor (exllama adds 1 during inference)
    izeros = izeros - 1
    # exllama uses row packing for weights and column packing for zeros
    qzeros = pack(izeros, direction="column")
    qweight = pack(iweights, direction="row")

    return qweight, qzeros

def pseudo_quantize_tensor(group_size, zero_point, w_bit, w: torch.Tensor):
    org_w_shape = w.shape
    if group_size > 0:
        assert org_w_shape[-1] % group_size == 0, f"org_w_shape ({org_w_shape[-1]}) must be a multiple of group_size ({group_size})!"
        w = w.reshape(-1, group_size)
    assert w.dim() == 2
    assert torch.isnan(w).sum() == 0

    # zero point quantization
    if zero_point:
        max_val = w.amax(dim=1, keepdim=True)
        min_val = w.amin(dim=1, keepdim=True)
        max_int = 2**w_bit - 1
        min_int = 0
        scales = (max_val - min_val).clamp(min=1e-5) / max_int
        zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)
        w = (
            torch.clamp(torch.round(w / scales) + zeros, min_int, max_int) - zeros
        ) * scales
        zeros = zeros.view(org_w_shape[0], -1)
    else:
        max_val = w.abs().amax(dim=1, keepdim=True)
        max_val = max_val.clamp(min=1e-5)
        max_int = 2 ** (w_bit - 1) - 1
        min_int = -(2 ** (w_bit - 1))
        scales = max_val / max_int
        zeros = None
        w = torch.clamp(torch.round(w / scales), min_int, max_int) * scales

    assert torch.isnan(scales).sum() == 0
    assert torch.isnan(w).sum() == 0
    scales = scales.view(org_w_shape[0], -1)
    w = w.reshape(org_w_shape)
    return w, scales, zeros

@torch.no_grad()
def compute_best_clip(
    group_size,
    zero_point, 
    w_bit,
    w: torch.Tensor,
    input_feat: torch.Tensor,
    n_grid=20,
    max_shrink=0.5,
    n_sample_token=512,
):
    assert w.dim() == 2
    org_w_shape = w.shape
    # w           [co, ci]      -> [co, 1, n_group, group size]
    # input_feat  [n_token, ci] -> [1, n_token, n_group, group size]
    group_size = group_size if group_size > 0 else org_w_shape[1]
    input_feat = input_feat.view(-1, input_feat.shape[-1])
    input_feat = input_feat.reshape(1, input_feat.shape[0], -1, group_size)

    # Compute input feature step size (minimum 1)
    step_size = max(1, input_feat.shape[1] // n_sample_token)
    input_feat = input_feat[:, ::step_size]
    
    w = w.reshape(org_w_shape[0], 1, -1, group_size)

    oc_batch_size = 2048 if org_w_shape[0] % 2048 == 0 else 64  # prevent OOM
    assert org_w_shape[0] % oc_batch_size == 0
    w_all = w
    best_max_val_all = []

    for i_b in range(org_w_shape[0] // oc_batch_size):
        w = w_all[i_b * oc_batch_size : (i_b + 1) * oc_batch_size]

        org_max_val = w.abs().amax(dim=-1, keepdim=True)  # co, 1, n_group, 1

        best_max_val = org_max_val.clone()
        min_errs = torch.ones_like(org_max_val) * 1e9
        input_feat = input_feat.to(w.device)
        org_out = (input_feat * w).sum(dim=-1)  # co, n_token, n_group

        for i_s in range(int(max_shrink * n_grid)):
            max_val = org_max_val * (1 - i_s / n_grid)
            min_val = -max_val
            cur_w = torch.clamp(w, min_val, max_val)
            q_w = pseudo_quantize_tensor(group_size, zero_point, w_bit, cur_w)[0]
            cur_out = (input_feat * q_w).sum(dim=-1)

            # co, 1, n_group, 1
            err = (cur_out - org_out).pow(2).mean(dim=1).view(min_errs.shape)
            del cur_w
            del cur_out
            cur_best_idx = err < min_errs
            min_errs[cur_best_idx] = err[cur_best_idx]
            best_max_val[cur_best_idx] = max_val[cur_best_idx]
        best_max_val_all.append(best_max_val)

    best_max_val = torch.cat(best_max_val_all, dim=0)
    return best_max_val.squeeze(1)

@torch.no_grad()
def search_best_clip(group_size, zero_point, w_bit, name, linear, input_feat):
    # due to qk bmm, it is hard to clip precisely
    max_val = compute_best_clip(
        group_size, zero_point, w_bit, linear.weight, input_feat
    )
    return (name, max_val)

def apply_quant(zero_point, version, w_bit, group_size, linear_layer: Dict[str, nn.Linear]):
        # NOTE: small regression in perplexity if linear layer uses .cpu().float()
        linear_layer = linear_layer.half()

        linear_layer.weight.data, scales, zeros = pseudo_quantize_tensor(
            group_size, zero_point, w_bit, linear_layer.weight.data
        )

        if version == "gemm":
            scales = scales.t().contiguous()
            if zeros is not None:
                zeros = zeros.t().contiguous()
            q_linear_module = WQLinear_GEMM

        elif version == "gemv":
            q_linear_module = WQLinear_GEMV

        elif version == "marlin":
            q_linear_module = WQLinear_Marlin

        elif version == "gemv_fast":
            q_linear_module = WQLinear_GEMVFast

        else:
            raise ValueError(f"Unknown version {version}")
        q_linear = q_linear_module.from_linear(
            linear=linear_layer,
            w_bit=w_bit,
            group_size=group_size,
            init_only=False,
            scales=scales,
            zeros=zeros,
        )
        return q_linear

@torch.no_grad()
def dq_fp8_weight(named_linears):
    logging.info(f'Start dq weight')
    raw_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    device = next(iter(named_linears.values())).weight.device
    with torch.cuda.device(device):
        for name, m in named_linears.items():
            if hasattr(m, "weight") and hasattr(m, "weight_scale_inv"):
                m.weight = weight_dequant(m.weight, m.weight_scale_inv)
            else:
                logging.warning(f"Fmt error: {name}")
    torch.set_default_dtype(raw_dtype)
    logging.info(f'Over dq')