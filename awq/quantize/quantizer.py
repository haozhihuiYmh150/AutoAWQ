import torch
import inspect
import logging
import functools
import torch.nn as nn
from tqdm import tqdm
from typing import Dict, List, Optional
from collections import defaultdict
from awq.utils.calib_data import get_calib_dataset
from awq.quantize.scale import apply_scale, apply_clip
from awq.utils.utils import (
    clear_memory, 
    get_best_device,
    timer,
)
from awq.utils.quant_utils import (
    pseudo_quantize_tensor,
    search_best_clip,
    apply_quant,
    dq_fp8_weight,
)
from awq.utils.module import (
    append_str_prefix,
    set_op_by_name,
    get_op_name,
    get_named_linears,
    exclude_layers_to_not_quantize,
)

class AwqExpertQuantizerDP(nn.Module):
    """
        Using Data Parallel at the module level.
        But here, expert's weight and input feature both is data, no weight :)
    """
    def __init__(self):
        super().__init__()
        # just for pass
        self.pass_w = nn.Parameter(torch.zeros(1), requires_grad=False)

    @classmethod
    def quant(cls, **kwargs):
        named_linears, input_feat = kwargs['named_linears'], kwargs['input_feat']
        # avoid broadcast input_feat
        class TensorWrapper:
            def __init__(self, obj):
                self._wrapped = obj
            def unwrap(self):
                return self._wrapped

        for name, feat in input_feat.items():
            input_feat[name] = TensorWrapper(feat)

        def split_dict(input_dict, n):
            keys = list(input_dict.keys())
            k, m = divmod(len(keys), n)
            return [keys[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]

        model = None
        device_ids=[i for i in range(torch.cuda.device_count())]
        def get_model():
            nonlocal model
            if model is None:
                model = cls()
                dp_model = torch.nn.DataParallel(model, device_ids=device_ids)
            return model, dp_model

        model, dp_model = get_model()
        model.to("cuda:0")
        model.tid2names = split_dict(named_linears, len(device_ids))
        model.tid2named_qlinears = [{} for _ in range(torch.cuda.device_count())]
        
        dp_model(**kwargs)

        named_qlinears = {}
        for sub_named_qlinears in model.tid2named_qlinears:
            named_qlinears.update(sub_named_qlinears)
        return named_qlinears

    def forward(self, is_apply_clip, version, module_name, group_size, export_compatible, zero_point, w_bit, module, named_linears, input_feat):
        t_id = torch.cuda.current_device()
        names = self.tid2names[t_id]

        for name in names:
            input_feat[name] = input_feat[name].unwrap().to("cuda:"+str(t_id))
            named_linears[name].to("cuda:"+str(t_id))

        # [STEP 3]: Compute and apply clipping list
        if is_apply_clip:
            with timer(desc=f'd{t_id}-search_best_clip'):
                avoid_clipping = ["q_", "k_", "query", "key", "Wqkv"]
                clip_list = [
                    search_best_clip(
                        group_size, zero_point, w_bit, name, named_linears[name], input_feat[name]
                    ) 
                    for name in names
                    if not any([_ in name for _ in avoid_clipping])
                ]
            with timer(desc=f'd{t_id}-apply_clip'):
                for name, max_val in clip_list:
                    apply_clip(module, name, max_val)
                clip_list = [
                    append_str_prefix(
                        clip_item, module_name + "."
                    )
                    for clip_item in clip_list
                ]
        # [STEP 4]: Quantize weights
        if not export_compatible:
            with timer(desc=f'd{t_id}-apply_quant'):
                named_qlinears = {}
                for name in names:
                    named_qlinears[name] = apply_quant(zero_point, version, w_bit, group_size, named_linears[name])
                for name, q_linear in named_qlinears.items():
                    q_linear.to("cuda:0")
                self.tid2named_qlinears[t_id] = named_qlinears

class AwqQuantizer:
    def __init__(
        self,
        awq_model,
        model,
        tokenizer,
        w_bit,
        group_size,
        zero_point,
        version,
        calib_data,
        split,
        text_column,
        duo_scaling,
        modules_to_not_convert=None,
        export_compatible=False,
        apply_clip=True,
        n_parallel_calib_samples=None,
        max_calib_samples=128,
        max_calib_seq_len=512,
        max_chunk_memory=1024 * 1024 * 1024,
    ) -> None:
        self.awq_model = awq_model
        self.model = model
        self.tokenizer = tokenizer
        self.w_bit = w_bit
        self.group_size = group_size
        self.zero_point = zero_point
        self.version = version
        self.calib_data = calib_data
        self.split = split
        self.text_column = text_column
        self.duo_scaling = duo_scaling
        self.export_compatible = export_compatible
        self.apply_clip = apply_clip
        self.n_parallel_calib_samples = n_parallel_calib_samples
        self.max_calib_samples = max_calib_samples
        self.max_calib_seq_len = max_calib_seq_len
        self.max_chunk_memory = max_chunk_memory
        self.modules_to_not_convert = (
            modules_to_not_convert if modules_to_not_convert is not None else []
        )
        self.modules, self.module_kwargs, self.inps = self.init_quant(
            n_samples=self.max_calib_samples, max_seq_len=self.max_calib_seq_len
        )

    def pseudo_dequantize_tensor(
        self, w: nn.Linear, scales: torch.Tensor, zeros: Optional[torch.Tensor] = None
    ):
        # get repeated count
        repeat_count = w.weight.data.shape[-1] // scales.shape[-1]
        scales = scales.repeat(1, repeat_count).reshape(w.weight.data.shape)

        # dequantize
        if self.zero_point:
            zeros = zeros.repeat(1, repeat_count).reshape(w.weight.data.shape)
            w = (w.weight.data - zeros) * scales
        else:
            w = w.weight.data * scales

        return w

    def quantize(self):
        for i in tqdm(range(len(self.modules)), desc="AWQ"):
            with timer(f'module {i} quant'):
                # Move module and inputs to correct device
                common_device = next(self.modules[i].parameters()).device
                if common_device is None or str(common_device) == "cpu":
                    best_device = get_best_device()
                    self.modules[i] = self.modules[i].to(best_device)
                    common_device = next(self.modules[i].parameters()).device

                if self.module_kwargs.get("position_ids") is not None:
                    self.module_kwargs["position_ids"] = self.module_kwargs[
                        "position_ids"
                    ].to(common_device)

                if self.module_kwargs.get("attention_mask") is not None:
                    self.module_kwargs["attention_mask"] = self.module_kwargs[
                        "attention_mask"
                    ].to(common_device)

                self.inps = self.inps.to(common_device)

                # We need to move the rotary embedding every time we move to a new module.
                # Transformers 4.45.0 moved rotary embedding to model definition as of this PR:
                # https://github.com/huggingface/transformers/pull/32617
                self.awq_model.move_embed(self.model, common_device)

                for k, v in self.module_kwargs.items():
                    # position embeddings found in tuple
                    if isinstance(v, tuple):
                        self.module_kwargs[k] = tuple(
                            item.to(common_device) if isinstance(item, (torch.Tensor, nn.Module)) 
                            else item for item in v
                        )

                # [STEP 1]: Get layer, extract linear modules, extract input features
                named_linears = get_named_linears(self.modules[i])

                # Filter out the linear layers we don't want to exclude
                named_linears = exclude_layers_to_not_quantize(
                    named_linears, self.modules_to_not_convert
                )

                input_feat = self._get_input_feat(self.modules[i], named_linears)

                dq_fp8_weight(named_linears)

                # [STEP 2]: Compute and apply scale list
                with timer(f'get_layers_for_scaling'):
                    module_config: List[Dict] = self.awq_model.get_layers_for_scaling(
                        self.modules[i], input_feat, self.module_kwargs
                    )
                    scales_list = [
                        self._search_best_scale(self.modules[i], **layer)
                        for layer in module_config
                    ]
                with timer(f'apply_scale'):
                    apply_scale(self.modules[i], scales_list, input_feat_dict=input_feat)
                    scales_list = append_str_prefix(
                        scales_list, get_op_name(self.model, self.modules[i]) + "."
                    )

                module_name = get_op_name(self.model, self.modules[i])

                if self.awq_model.model_type == "deepseek_v3":
                    with timer(desc=f'dp-state'):
                        named_qlinears = self.awq_ep_dp.quant(is_apply_clip=self.apply_clip, version=self.version, module_name=module_name, group_size=self.group_size, export_compatible=self.export_compatible, zero_point=self.zero_point, w_bit=self.w_bit, module=self.modules[i], named_linears=named_linears, input_feat=input_feat)
                else:
                    # [STEP 3]: Compute and apply clipping list
                    if self.apply_clip:
                        avoid_clipping = ["q_", "k_", "query", "key", "Wqkv"]
                        clip_list = [
                            search_best_clip(
                                self.group_size, self.zero_point, self.w_bit, name, named_linears[name], input_feat[name]
                            ) 
                            for name in named_linears.keys()
                            if not any([_ in name for _ in avoid_clipping])
                        ]
                        for name, max_val in clip_list:
                            apply_clip(self.modules[i], name, max_val)
                        clip_list = [
                            append_str_prefix(
                                clip_item, module_name + "."
                            )
                            for clip_item in clip_list
                        ]
                    # [STEP 4]: Quantize weights
                    if not self.export_compatible:
                        named_qlinears = {
                            name: apply_quant(self.zero_point, self.version, self.w_bit, self.group_size, linear)
                            for name, linear in named_linears.items()
                        }
                if not self.export_compatible:
                    for name, q_linear in named_qlinears.items():
                        set_op_by_name(self.modules[i], name, q_linear)

                self.modules[i].to("cpu") # offload layer
                clear_memory()

    # def pack(self):
    #     for i in tqdm(range(len(self.modules)), desc="Packing"):
    #         named_linears = get_named_linears(self.modules[i])
    #         named_linears = exclude_layers_to_not_quantize(
    #             named_linears, self.modules_to_not_convert
    #         )
    #         apply_quant(self.zero_point, self.version, self.w_bit, self.group_size, named_linears)
    #         clear_memory()

    @torch.no_grad()
    def _module_forward(
        self, x: torch.Tensor, module: torch.nn.Module, module_kwargs: Dict
    ) -> torch.Tensor:
        if self.n_parallel_calib_samples is None:
            # runs through all samples at once
            module_output = module(x, **module_kwargs)
            if isinstance(module_output, tuple):
                module_output = module_output[0]
        else:
            # memory efficiently runs through all calibration samples
            # but only n_parallel_calib_samples at a time
            module_output = []
            partitioned_inputs = torch.split(x, self.n_parallel_calib_samples)
            for x_partial in partitioned_inputs:
                partial_output = module(x_partial, **module_kwargs)

                if isinstance(partial_output, tuple):
                    partial_output = partial_output[0]

                module_output.append(partial_output.clone())

            module_output = torch.cat(module_output, dim=0)

        return module_output

    @torch.no_grad()
    def _search_best_scale(
        self,
        module,
        prev_op,
        layers: List[nn.Linear],
        inp: torch.Tensor,
        module2inspect=None,
        kwargs={},
    ):
        if module2inspect is None:
            assert len(layers) == 1
            module2inspect = layers[0]

        if "use_cache" in kwargs:
            kwargs.pop("use_cache")

        # Put x on the right device
        inp = inp.to(next(module2inspect.buffers()).device)

        # [STEP 1]: Compute per-channel mean of normalised weights
        # All layer weights are concatted together
        weight = torch.cat([_m.weight for _m in layers], dim=0)
        org_shape = weight.shape
        # The weights are reshaped to be organised by quantization group
        weight = weight.view(-1, self.group_size)
        # Calculates the relative magnitude of the weights within each of the quantization groups,
        # and rescales each group individually so that each group has weights on a 0-1 scale.
        w_scale = weight.abs() / (weight.abs().amax(dim=1, keepdim=True) + 1e-6)
        # Resizes the rescaled weight matrix back up to its original dimensions
        w_scale = w_scale.view(org_shape)
        # Gets the average rescaled magnitude for each output channel
        w_mean = w_scale.mean(0)

        # [STEP 2]: Compute per-channel mean of the input activation with chunking
        # move inp to cpu to avoid memory leak
        inp_flat = inp.abs().view(-1, inp.shape[-1])
        num_elements = inp_flat.size(0)
        num_channels = inp_flat.size(1)
        element_size_bytes = inp_flat.element_size() * 2 # multiplied by 2 for FP32

        # Calculate chunk size dynamically based on max_chunk_memory
        chunk_size = int(self.max_chunk_memory // (element_size_bytes * num_channels))
        chunk_size = min(chunk_size, num_elements)

        # Use float32 for sum calculation
        x_sum = torch.zeros(num_channels, dtype=torch.float32, device=inp.device)
        
        for i in range(0, num_elements, chunk_size):
            end = min(i + chunk_size, num_elements)
            chunk_sum = inp_flat[i:end].to(torch.float32).sum(dim=0)
            x_sum += chunk_sum.to(inp.device)

        x_mean = (x_sum / num_elements).to(inp.dtype)

        # [STEP 3]: Compute output of module
        with torch.no_grad():
            module_kwargs = self._sanitize_kwargs(kwargs, module2inspect)
            fp16_output = self._module_forward(inp, module2inspect, module_kwargs)
            fp16_output = fp16_output.clip(torch.finfo(fp16_output.dtype).min, torch.finfo(fp16_output.dtype).max)

        # [STEP 4]: Compute loss
        best_scales = self._compute_best_scale(
            inp, w_mean, x_mean, module2inspect, layers, fp16_output, module_kwargs
        )

        return (
            get_op_name(module, prev_op),
            tuple([get_op_name(module, m) for m in layers]),
            best_scales,
        )

    def _compute_best_scale(
        self,
        x: torch.Tensor,
        w_mean: torch.Tensor,
        x_mean: torch.Tensor,
        module2inspect: torch.nn.Module,
        linears2scale: List[nn.Linear],
        fp16_output: torch.Tensor,
        kwargs: Dict={},
    ):
        """
        Compute loss and select best scales

        L(s) = || Q(W * s) (s^-1 * X) - W * X ||
        Q: weight quantization function | pseudo_quantize_tensor(W * s)
        X: inputs from calib dataset    | X
        W: original weights in FP16     | layer
        s: per channel scaling factor   | s^-1 * X
        """
        n_grid = 20
        history = []
        best_ratio = -1
        best_scales = None
        best_error = float("inf")

        org_sd = {k: v.clone() for k, v in module2inspect.state_dict().items()}

        device = x.device
        x_mean = x_mean.view(-1).to(device)
        w_mean = w_mean.view(-1).to(device)

        for ratio in range(n_grid):
            # create new scales
            ratio = ratio / n_grid

            # NOTE: s^-1 * x is fused here, according to paper
            if self.duo_scaling:
                scales = (x_mean.pow(ratio) / (w_mean.pow(1 - ratio) + 1e-4)).clamp(min=1e-4)
            else:
                scales = x_mean.pow(ratio).clamp(min=1e-4).view(-1)
            scales = scales / (scales.max() * scales.min()).sqrt()
            scales_view = scales.view(1, -1).to(device)

            # avoid scaling values that overflow
            scales[torch.isinf(scales)] = 1
            scales[torch.isnan(scales)] = 1

            # Q(W * s)
            for fc in linears2scale:
                fc.weight.mul_(scales_view)
                fc.weight.data = (
                    pseudo_quantize_tensor(self.group_size, self.zero_point, self.w_bit, fc.weight.data)[0] / scales_view
                )

            # W * X
            int_w_output = self._module_forward(x, module2inspect, kwargs)
            int_w_output = int_w_output.clip(torch.finfo(int_w_output.dtype).min, torch.finfo(int_w_output.dtype).max)

            # compute mean squared error (L2 norm)
            loss = self._compute_loss(fp16_output, int_w_output, device)

            history.append(loss)
            if loss < best_error:
                best_error = loss
                best_ratio = ratio
                best_scales = scales.clone()
            module2inspect.load_state_dict(org_sd)

        if best_ratio == -1:
            logging.debug(history)
            raise Exception

        assert torch.isnan(best_scales).sum() == 0, best_scales

        return best_scales.detach()

    @torch.no_grad()
    def _compute_loss(
        self,
        fp16_output: torch.Tensor,
        int_w_output: torch.Tensor,
        device: torch.device,
    ):
        loss = 0.0
        fp16_output_flat = fp16_output.view(-1)
        int_w_output_flat = int_w_output.view(-1)
        num_elements = fp16_output_flat.size(0)
        element_size_bytes = fp16_output.element_size()

        # Calculate chunk size dynamically based on max_chunk_memory
        # Divide the max_chunk_memory by twice the element size
        chunk_size = self.max_chunk_memory // (element_size_bytes * 2)
        chunk_size = min(chunk_size, num_elements)

        # Split the computation into chunks
        fp16_chunks = torch.split(fp16_output_flat, chunk_size)
        int_w_chunks = torch.split(int_w_output_flat, chunk_size)

        # Compute the loss for each chunk
        for fp16_chunk, int_w_chunk in zip(fp16_chunks, int_w_chunks):
            chunk_loss = (fp16_chunk.to(device) - int_w_chunk.to(device)).float().pow(2).sum().item()
            loss += chunk_loss

        # Normalize the loss by the total number of elements
        loss /= num_elements

        return loss

    def init_quant(self, n_samples=128, max_seq_len=512):
        modules = self.awq_model.get_model_layers(self.model)
        samples = get_calib_dataset(
            data=self.calib_data,
            tokenizer=self.tokenizer,
            n_samples=n_samples,
            max_seq_len=max_seq_len,
            split=self.split,
            text_column=self.text_column,
        )
        samples = torch.cat(samples, dim=0)

        inps = []
        layer_kwargs = {}

        best_device = get_best_device()
        modules[0] = modules[0].to(best_device)
        self.awq_model.move_embed(self.model, best_device)

        # get input and kwargs to layer 0
        # with_kwargs is only supported in PyTorch 2.0
        # use this Catcher hack for now
        class Catcher(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def forward(self, *args, **kwargs):
                # assume first input to forward is hidden states
                if len(args) > 0:
                    hidden_states = args[0]
                    del args
                else:
                    first_key = list(kwargs.keys())[0]
                    hidden_states = kwargs.pop(first_key)

                inps.append(hidden_states)
                layer_kwargs.update(kwargs)
                raise ValueError  # early exit to break later inference

        # patch layer 0 to catch input and kwargs
        modules[0] = Catcher(modules[0])
        try:
            self.model(samples.to(next(self.model.parameters()).device))
        except ValueError:  # work with early exit
            pass
        modules[0] = modules[0].module  # restore

        # Update the layer kwargs with `prepare_inputs_for_generation` method
        # that takes care of everything to avoid unexpected errors.
        layer_kwargs = self.model.prepare_inputs_for_generation(samples, **layer_kwargs)
        # Pop the input_ids as they are not needed at all.
        layer_kwargs.pop("input_ids")

        del samples
        inps = inps[0]

        if layer_kwargs.get("attention_mask") is not None:
            layer_kwargs["attention_mask"] = layer_kwargs["attention_mask"].to(
                best_device
            )
        if self.awq_model.model_type == "deepseek_v3":
            self.awq_ep_dp = AwqExpertQuantizerDP

        return modules, layer_kwargs, inps

    def _get_input_feat(self, layer, named_linears):
        # firstly, get input features of all linear layers
        def cache_input_hook(m, x, y, name, feat_dict):
            x = x[0]
            x = x.detach()
            feat_dict[name].append(x)

        input_feat = defaultdict(list)
        handles = []

        # FIXME: Workaround for Mixtral to use block_sparse_moe input features
        if self.awq_model.model_type == "mixtral":
            named_linears = {
                **named_linears,
                "block_sparse_moe": layer.block_sparse_moe,
            }

        if self.awq_model.model_type == "deepseek_v2" or self.awq_model.model_type == "deepseek_v3":
            named_linears = {
                **named_linears,
                "mlp": layer.mlp,
            }

        for name in named_linears:
            handles.append(
                named_linears[name].register_forward_hook(
                    functools.partial(cache_input_hook, name=name, feat_dict=input_feat)
                )
            )
        self.inps = self.inps.to(next(layer.parameters()).device)  # in case multi-gpu
        # get output as next layer's input

        # Sanitize the kwargs in case we use transformers version that contains
        # kwargs that are not handled by the module.
        # Useful for trust_remote_code models.
        module_kwargs = self._sanitize_kwargs(self.module_kwargs, layer)

        self.inps = self._module_forward(self.inps, layer, module_kwargs)
        for h in handles:
            h.remove()
        # now solve for scaling and clipping
        input_feat = {k: torch.cat(v, dim=0) for k, v in input_feat.items()}

        return input_feat

    def _sanitize_kwargs(self, inputs_kwargs, module):
        """
        Remove the arguments that are not supported in the module's
        forward pass to avoid breaking behaviour between different versions
        of transformers.

        Args:
            inputs_kwargs (`dict`):
                The input dictionary to pass to the model layer
            module (`torch.nn.Module`):
                Target module to quantize.
        """
        module_signature = inspect.signature(module.forward).parameters
        sanitized_kwargs = {}
        for k, v in inputs_kwargs.items():
            if k in module_signature:
                sanitized_kwargs[k] = v
        return sanitized_kwargs
