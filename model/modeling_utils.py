#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2026/05/15 11:42
# @Author  : XinYi Huang
# @FileName: modeling_utils.py
# @Software: Trae CN
import torch
from torch import nn
from torch.nn import functional as F

from flash_attn import flash_attn_with_kvcache, flash_attn_varlen_func
from transformers.cache_utils import DynamicCache

from typing import Optional, Dict


def repeat_kv(
        hidden_states: torch.Tensor,
        n_reps: int
):
    if n_reps == 1:
        return hidden_states

    return torch.repeat_interleave(hidden_states, n_reps, dim=1)


def eager_attention_forward(
        module: nn.Module,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        dropout: float = 0.0,
        scaling: float = None,
        **kwargs
):
    if hasattr(module, "num_kv_groups"):
        k = repeat_kv(k, module.num_kv_groups)
        v = repeat_kv(v, module.num_kv_groups)

    q = q * scaling
    attn_weights = q @ k.transpose(2, 3)

    if attn_mask is not None:
        attn_weights = attn_weights + attn_mask

    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
    attn_weights = F.dropout(attn_weights, dropout, training=module.training)

    attn_output = attn_weights @ v
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


def sdpa_attention_forward(
        module: nn.Module,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        dropout: float = 0.0,
        scaling: float = None,
        **kwargs
):
    if hasattr(module, "num_kv_groups"):
        k = repeat_kv(k, module.num_kv_groups)
        v = repeat_kv(v, module.num_kv_groups)

    is_causal = q.shape[2] > 1 and module.is_causal

    attn_output = F.scaled_dot_product_attention(
        q,
        k,
        v,
        is_causal=is_causal,
        attn_mask=attn_mask,
        dropout_p=dropout,
        scale=scaling
    )

    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, None


def _get_unpad_data(attn_mask: torch.Tensor):
    seqlens_in_batch = attn_mask.sum(-1, dtype=torch.int32)
    indices = attn_mask.flatten().nonzero().squeeze(1)
    max_seqlen_in_batch = seqlens_in_batch.max()
    cu_seqlens = F.pad(seqlens_in_batch.cumsum(0, dtype=torch.int32), (1, 0), value=0)

    return indices, cu_seqlens, max_seqlen_in_batch


def _unpad_input(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: torch.Tensor,
):
    indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attn_mask)

    q_len = q.shape[1]
    k_len = k.shape[1]

    k = k.flatten(0, 1)[indices_k]
    v = v.flatten(0, 1)[indices_k]

    if q_len == k_len:
        q = q.flatten(0, 1)[indices_k]
        indices_q = indices_k
        cu_seqlens_q = cu_seqlens_k
        max_seqlen_in_batch_q = max_seqlen_in_batch_k

    elif q_len < k_len:
        q = q.flatten(0, 1)
        max_seqlen_in_batch_q = q_len
        cu_seqlens_q = torch.arange(0, q.shape[0] + 1, q_len, device=q.device, dtype=torch.int32)
        indices_q = torch.arange(q.shape[0], device=q.device)
    else:
        raise NotImplementedError("q_length < kv_length is not implemented yet.")

    return (
        q,
        k,
        v,
        indices_q,
        (cu_seqlens_q, cu_seqlens_k),
        (max_seqlen_in_batch_q, max_seqlen_in_batch_k)
    )


def _pad_output(
        hidden_states: torch.Tensor,
        indices: torch.Tensor,
        batch: int,
        seqlen: int
):
    output = torch.zeros(
        batch * seqlen, *hidden_states.shape[1:], device=hidden_states.device, dtype=hidden_states.dtype
    )
    
    output.scatter_(
        0, 
        indices.view(-1, 1, 1).expand(-1, *hidden_states.shape[1:],),
        hidden_states
    )

    output = output.reshape(batch, seqlen, *hidden_states.shape[1:])

    return output


def flash_attention_forward(
        module: nn.Module,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: torch.Tensor,
        dropout: float = 0.0,
        scaling: float = None,
        **kwargs
):
    if hasattr(module, "num_kv_groups"):
        k = repeat_kv(k, module.num_kv_groups)
        v = repeat_kv(v, module.num_kv_groups)

    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    batch, seqlen = q.shape[:2]
    is_causal = module.is_causal

    (
        q,
        k,
        v,
        indices_q,
        (cu_seqlens_q, cu_seqlens_k),
        (max_seqlen_in_batch_q, max_seqlen_in_batch_k)
    ) = _unpad_input(
        q,
        k,
        v,
        attn_mask
    )

    out_unpad = flash_attn_varlen_func(
        q.to(torch.bfloat16),
        k.to(torch.bfloat16),
        v.to(torch.bfloat16),
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_in_batch_q,
        max_seqlen_in_batch_k,
        dropout_p=dropout,
        softmax_scale=scaling,
        causal=is_causal
    ).to(q.dtype)

    if isinstance(out_unpad, tuple):
        out_unpad = out_unpad[0]

    out = _pad_output(
        out_unpad,
        indices_q,
        batch,
        seqlen
    )

    return out, None


ALL_ATTENTION_FUNCTIONS: Dict = {
        "flash_attention_2": flash_attention_forward,
        "eager": eager_attention_forward,
        "sdpa": sdpa_attention_forward
    }

ACT2FN: Dict = {
    "gelu": nn.GELU(),
    "silu": nn.SiLU(),
    "tanh": nn.Tanh(),
    "relu": nn.ReLU(),
    "leaky_relu": nn.LeakyReLU()
}

def rotate_half(x: torch.Tensor):
    """implemented by qwen3"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        unsqueeze_dim: int = 1
):
    """implemented by qwen3"""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
