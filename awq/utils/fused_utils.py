import torch

from awq.modules.linear import (
    WQLinear_GEMM,
    WQLinear_GEMV,
    WQLinear_Marlin,
    WQLinear_Exllama,
    WQLinear_ExllamaV2,
    WQLinear_GEMVFast,
)


def prepare_cache(blocks, seqlen: int) -> int:
    for block in blocks:
        start_pos = block.attn.start_pos
        will_cache_be_exceeded = start_pos + seqlen > block.attn.max_seq_len

        # Reset and avoid retaining state when processing context
        if seqlen > 1 and (will_cache_be_exceeded or start_pos > 0):
            block.attn.start_pos = block.attn.cache.roll_kv_n_steps(
                start_pos, n=start_pos
            )

        # Slowly roll out old tokens without performance hit if exceeded during decoding
        elif seqlen == 1 and will_cache_be_exceeded:
            block.attn.start_pos = block.attn.cache.roll_kv_n_steps(start_pos, n=100)


def prepare_input_ids(input_ids: torch.Tensor, last_forward_num_tokens: int):
    # NOTE: from transformers 4.35.0, input_ids includes full context during decoding
    num_input_tokens = input_ids.shape[-1]
    num_new_tokens = num_input_tokens

    if num_input_tokens != 1:
        num_new_tokens = num_input_tokens - last_forward_num_tokens

        # after context is processed, slice to latest token
        if num_new_tokens == 1:
            input_ids = input_ids[:, -1:]

    return input_ids, last_forward_num_tokens + num_new_tokens

def get_attention_shapes(
    attention_shapes, n_heads, n_kv_heads, head_dim
):
    if attention_shapes is not None:
        attention_shapes = attention_shapes

    elif n_kv_heads == 0:
        attention_shapes = {
            "xqkv_view": (-1, n_heads, head_dim),
            "xq_slice": lambda xqkv: xqkv[:, :, 0],
            "xk_slice": lambda xqkv: xqkv[:, :, 1],
            "xv_slice": lambda xqkv: xqkv[:, :, 2],
            "xq_view": (n_heads, head_dim),
            "xk_view": (n_heads, head_dim),
            "xv_view": (n_heads, head_dim),
            "xk_reshape": (n_heads, head_dim // 8, 8),
            "single_xq_view": (n_heads, head_dim),
            "single_xk_view": (n_heads, head_dim),
            "single_xv_view": (n_heads, head_dim),
        }

    else:
        attention_shapes = {
            "xqkv_view": (n_heads + n_kv_heads * 2, head_dim),
            "xq_slice": lambda xqkv: xqkv[:, :, 0:n_heads],
            "xk_slice": lambda xqkv: xqkv[:, :, n_heads : (n_heads + n_kv_heads)],
            "xv_slice": lambda xqkv: xqkv[:, :, -n_kv_heads:],
            "xq_view": (n_heads, head_dim),
            "xk_view": (n_kv_heads, head_dim),
            "xv_view": (n_kv_heads, head_dim),
            "xk_reshape": (n_kv_heads, head_dim // 8, 8),
            "single_xq_view": (n_heads, head_dim),
            "single_xk_view": (n_kv_heads, head_dim),
            "single_xv_view": (n_kv_heads, head_dim),
        }

    return attention_shapes
