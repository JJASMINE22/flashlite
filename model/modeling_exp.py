#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2026/04/27 08:35
# @Author  : XinYi Huang
# @FileName: modeling_exp.py
# @Software: Trae CN
import torch
from torch import nn
from torch.nn import functional as F

from transformers.cache_utils import Cache, DynamicCache

from configuration.model_config import FlashLiteMLAConfig
from .modeling import RMSNorm, apply_rotary_pos_emb
from .modeling_utils import ALL_ATTENTION_FUNCTIONS

from typing import Tuple, Optional

# multi latent attention
class MLA(nn.Module):
    
    def __init__(
        self,
        config: FlashLiteMLAConfig,
        layer_idx: int
    ):
        super().__init__()
        assert config.num_attention_heads >= config.num_key_value_heads
        assert config.num_attention_heads % config.num_key_value_heads == 0
        assert config.qk_head_dim >= config.v_head_dim

        self.config = config
        self.layer_idx = layer_idx

        self.num_kv_groups = config.num_attention_heads // config.num_key_value_heads
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.attn_dropout = config.attn_dropout
        self.scaling = config.qk_head_dim ** -.5
        
        self.q_lora_rank = config.q_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_head_dim = config.qk_head_dim
        self.v_head_dim = config.v_head_dim
        self.is_causal = True

        # define query projection
        self.q_a_proj = nn.Linear(
            config.hidden_size, 
            self.q_lora_rank, 
            bias=config.attn_bias
        )
        self.q_a_norm = RMSNorm(self.q_lora_rank, config.rms_norm_eps)
        self.q_b_proj = nn.Linear(
            self.q_lora_rank, 
            self.num_attention_heads * self.qk_head_dim, 
            bias=False
        )

        # define key value projection
        self.kv_a_proj_with_mqa = nn.Linear(
            config.hidden_size, 
            self.kv_lora_rank + self.qk_rope_head_dim, 
            bias=config.attn_bias
        )
        self.kv_a_norm = RMSNorm(self.kv_lora_rank, config.rms_norm_eps)
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank, 
            self.num_key_value_heads * (self.qk_nope_head_dim + self.v_head_dim), 
            bias=False
        )

        # define output projection
        self.o_proj = nn.Linear(
            self.num_attention_heads * self.v_head_dim, 
            config.hidden_size,
            bias=config.attn_bias
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache],
        cache_position: Optional[torch.Tensor],
        **kwargs
    ):
        B, N, _ = hidden_states.shape
        query_shape = (B, N, -1, self.qk_head_dim)
        key_value_shape = (B, N, -1, self.qk_nope_head_dim + self.v_head_dim)

        q_states = self.q_b_proj(self.q_a_norm(self.q_a_proj(hidden_states)))
        q_states = q_states.view(query_shape).transpose(1, 2)
        q_pass, q_rot = torch.split(q_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        kv_compressed = self.kv_a_proj_with_mqa(hidden_states)
        kv_pass, k_rot = torch.split(kv_compressed, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_rot = k_rot.unsqueeze(1)

        kv_pass = self.kv_b_proj(self.kv_a_norm(kv_pass)).view(key_value_shape).transpose(1, 2)
        k_pass, v_states = torch.split(kv_pass, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        cos, sin = position_embeddings
        q_rot, k_rot = apply_rotary_pos_emb(q_rot, k_rot, cos, sin)
        k_rot = k_rot.expand(*k_pass.shape[:-1], -1)

        q_states = torch.cat([q_pass, q_rot], dim=-1)
        k_states = torch.cat([k_pass, k_rot], dim=-1)

        if past_key_values is not None:
            cache_kwargs = {"cos": cos, "sin": sin, "cache_position": cache_position}
            k_states, v_states = past_key_values.update(q_states, k_states, self.layer_idx, cache_kwargs)
        
        # make sure query | key | value have the same head dimension in flash attention 2
        if self.config._attn_implementation == "flash_attention_2" and self.qk_head_dim != self.v_head_dim:
            v_states = F.pad(v_states, [0, self.qk_head_dim - self.v_head_dim])

        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        o_states, attn_weights = attention_interface(
            self, q_states, k_states, v_states, attention_mask, dropout=self.attn_dropout, scaling=self.scaling, **kwargs
        )

        if self.config._attn_implementation == "flash_attention_2" and self.qk_head_dim != self.v_head_dim:
            o_states = o_states[:, :, :, :self.v_head_dim]

        o_states = o_states.reshape(B, N, -1)
        o = self.o_proj(o_states)

        return o, attn_weights

# linear attention cache layer
class DynamicLinearCacheLayer:
    def __init__(self):
        self.conv_state: torch.Tensor | None = None
        self.recurrent_state: torch.Tensor | None = None
        self.is_conv_state_initialized: bool = False
        self.is_recurrent_state_initialized: bool = False

    def lazy_initialization(
        self,
        conv_state: torch.Tensor | None = None,
        recurrent_state: torch.Tensor | None = None,
    ):
        f'''
        Lazy initialization of the cache layer.
        :param conv_state: Convolutional states, shape as (B, D, conv_kernel_size).
        :param recurrent_state: Recurrent states, shape as (B, H, D_k, D_v).
        '''
        if conv_state is not None:
            self.dtype, self.device = conv_state.dtype, conv_state.device
            self.batch_size, self.conv_kernel_size = conv_state.shape[0], conv_state.shape[2]

            self.conv_state = torch.zeros_like(conv_state, device=self.device, dtype=self.dtype)
            self.is_conv_state_initialized = True

        if recurrent_state is not None:
            self.dtype = recurrent_state.dtype if not hasattr(self, "dtype") else self.dtype
            self.device = recurrent_state.device if not hasattr(self, "device") else self.device

            self.recurrent_state = torch.zeros_like(recurrent_state, device=self.device, dtype=self.dtype)

            self.is_recurrent_state_initialized = True


    def update(
        self,
        states: torch.Tensor | None = None
    ):
        if not self.is_states_initialized:
            self.lazy_initialization(states)

        if self.is_states_initialized:
            with torch.autocast(
                enabled=False, device_type=self.device.type
            ):
                states = states.float() + self.states

            self.states = states[:, -1:, ...]

            return states.to(dtype=self.dtype, device=self.device)

    def update_conv_state(
        self,
        conv_state: torch.Tensor | None = None
    ):
        if not self.is_conv_state_initialized:
            self.lazy_initialization(conv_state=conv_state)

        if self.is_conv_state_initialized:
            self.conv_state.copy_(conv_state[..., -self.conv_kernel_size:])

    def update_recurrent_state(
        self,
        recurrent_state: torch.Tensor | None = None
    ):
        if not self.is_recurrent_state_initialized:
            self.lazy_initialization(recurrent_state=recurrent_state)

        if self.is_recurrent_state_initialized:
            self.recurrent_state.copy_(recurrent_state)

    def get_kernel_size(self):
        if not self.is_conv_state_initialized:
            return 0
        return self.conv_kernel_size

    def offload(self):
        self.conv_state = self.conv_state.to(dtype=self.dtype, device="cpu")
        self.recurrent_state = self.recurrent_state.to(dtype=self.dtype, device="cpu")

    def reset(self):
        self.conv_state = None
        self.recurrent_state = None
        self.is_conv_state_initialized = False
        self.is_recurrent_state_initialized = False


class Cache:
    def __init__(
        self,
        layers: list[DynamicLinearCacheLayer] | None = None,
        layer_class_to_relicate: type[DynamicLinearCacheLayer] | None = None,
        offloading: bool = False
    ):

        self.layers = layers if layers is not None else []
        self.layer_class_to_relicate = layer_class_to_relicate
        self.offloading = offloading

    def update(
        self,
        layer_idx: int,
        states: torch.Tensor
    ):
        if self.layer_class_to_relicate is not None:
            while len(self.layers) <= layer_idx:
                layer = self.layer_class_to_relicate()
                self.layers.append(layer)

        states = self.layers[layer_idx].update(states)

        if self.offloading:
            self.offload(layer_idx)

        return states

    def update_conv_state(
        self,
        layer_idx: int,
        conv_state: torch.Tensor | None = None
    ):
        if len(self.layers) <= layer_idx:
            raise ValueError(f"Layer index {layer_idx} is out of range. Cache has {len(self.layers)} layers.")
        self.layers[layer_idx].update_conv_state(conv_state)

    def update_recurrent_state(
        self,
        layer_idx: int,
        recurrent_state: torch.Tensor | None = None
    ):
        if len(self.layers) <= layer_idx:
            raise ValueError(f"Layer index {layer_idx} is out of range. Cache has {len(self.layers)} layers.")
        self.layers[layer_idx].update_recurrent_state(recurrent_state)

    def get_kernel_size(
        self,
        layer_idx: int
    ):
        if len(self.layers) <= layer_idx:
            return 0
        return self.layers[layer_idx].get_kernel_size()

    def offload(
        self,
        layer_idx: int
    ):
        if len(self.layers) <= layer_idx:
            return
        self.layers[layer_idx].offload()

    def reset(
        self,
        layer_idx: int
    ):
        if len(self.layers) <= layer_idx:
            return
        self.layers[layer_idx].reset()


class RMSNormGated(nn.RMSNorm):

    def forward(
        self,
        hidden_states: torch.Tensor,
        gated: torch.Tensor,
    ):
        hidden_states = super(RMSNormGated, self).forward(hidden_states)
        hidden_states = hidden_states * F.silu(gated)

        return hidden_states


def apply_mask_to_padding_state(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
):
    '''
    Apply mask to padding hidden states.
    Args:
        hidden_states: hidden states, shape as (batch_size, hidden_size, seq_len).
        attention_mask(torch.Tensor | None, default=None): attention mask, shape as (batch_size, seq_len).
    Returns:
        Output hidden states.
    '''

    # apply mask to padding hidden states(except token by token inference)
    if attention_mask is not None and hidden_states.shape[1] > 1:
        while attention_mask.ndim < hidden_states.ndim:
            attention_mask = attention_mask.unsqueeze(-1)
        hidden_states = hidden_states * attention_mask

    return hidden_states



def torch_causal_conv1d_update(
    hidden_states: torch.Tensor,
    conv_state: torch.Tensor | None,
    weight: nn.Parameter
):
    '''
    Causal 1D convolution update for gated delta net.
    Args:
        hidden_states: Input hidden states, shape as (batch_size, hidden_size, seq_len).
        conv_state: Convolution state, shape as (batch_size, hidden_size, conv_kernel_size).
        weight: Convolution weight, shape as (hidden_size, 1, conv_kernel_size).
    Returns:
        Output hidden states.
    '''

    _, hidden_size, seq_len = hidden_states.shape
    if conv_state is not None:
        hidden_states = torch.cat(
            (conv_state, hidden_states),
            dim=-1
        )
    hidden_states = F.conv1d(
        hidden_states,
        weight,
        groups=hidden_size
    )
    # cause conv state's seq_len is conv_kernel_size, so we need to crop it
    out = F.silu(hidden_states[..., -seq_len:])

    return out.to(hidden_states.dtype)


def fix_query_key_value_ordering(
    mixed_qkvz: torch.Tensor,
    mixed_ba: torch.Tensor,
    num_value_heads: int,
    num_key_heads: int,
    value_head_dim: int,
    key_head_dim: int,
):
    '''
    Fix query key value ordering in mixed qkvz and ba.
    Args:
        mixed_qkvz: Mixed qkvz, shape as (batch_size, hidden_size, seq_len).
        mixed_ba: Mixed ba, shape as (batch_size, hidden_size, seq_len).
        num_value_heads: Number of value heads.
        num_key_heads: Number of key heads.
        value_head_dim: Dimension of value head.
        key_head_dim: Dimension of key head.
    Returns:
        Fixed query, key, value, z, b, a.
    '''
    new_mixed_qkvz_shape = mixed_qkvz.shape[:-1] + (
        num_key_heads,
        (key_head_dim * 2 + value_head_dim * num_value_heads // num_key_heads * 2),
    )
    new_mixed_ba_shape = mixed_ba.shape[:-1] + (
        num_key_heads,
        2 * num_value_heads // num_key_heads,
    )
    mixed_qkvz = mixed_qkvz.reshape(new_mixed_qkvz_shape)
    mixed_ba = mixed_ba.reshape(new_mixed_ba_shape)

    split_args_list_qkvz = [
        key_head_dim,
        key_head_dim,
        value_head_dim * num_value_heads // num_key_heads,
        value_head_dim * num_value_heads // num_key_heads
    ]
    split_args_list_ba = [
        num_value_heads // num_key_heads,
        num_value_heads // num_key_heads
    ]

    query, key, value, z = torch.split(
        mixed_qkvz,
        split_args_list_qkvz,
        dim=-1
    )
    b, a = torch.split(
        mixed_ba,
        split_args_list_ba,
        dim=-1
    )

    value = value.reshape(*value.size()[:2], -1, value_head_dim)
    z = z.reshape(*z.size()[:2], -1, value_head_dim)
    b = b.reshape(*b.size()[:2], -1)
    a = a.reshape(*a.size()[:2], -1)

    return query, key, value, z, b, a


def torch_recurrent_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_recurrent_state: torch.Tensor | None = None,
    qk_norm: bool = False
):
    '''
    Recurrent gated delta rule calculated by torch.
    Args:
        query: Query, shape as (batch_size, seq_len, hidden_size).
        key: Key, shape as (batch_size, seq_len, hidden_size).
        value: Value, shape as (batch_size, seq_len, hidden_size).
        g: Gate, shape as (batch_size, seq_len, num_value_heads).
        beta: Beta, shape as (batch_size, seq_len, num_value_heads).
        initial_recurrent_state: Initial recurrent state, shape as (batch_size, num_value_heads, key_head_dim, value_head_dim).
        qk_norm: Whether to normalize query key.
    Returns:
        attention_output, shape as (batch_size, seq_len, num_value_heads, value_head_dim).
        last_recurrent_state, shape as (batch_size, num_value_heads, key_head_dim, value_head_dim).
    '''

    if qk_norm:
        query = F.normalize(query, dim=-1, eps=1e-6)
        key = F.normalize(key, dim=-1, eps=1e-6)

    batch_size, sequence_len, num_heads, key_head_dim = key.shape
    value_head_dim = value.shape[-1]

    last_recurrent_state = initial_recurrent_state if initial_recurrent_state is not None else torch.zeros(
        batch_size, num_heads, key_head_dim, value_head_dim,
        device=query.device, dtype=query.dtype
    )
    attn_out = torch.zeros(
        batch_size, sequence_len, num_heads, value_head_dim,
        device=query.device, dtype=query.dtype
    )

    for i in range(sequence_len):
        q_t = query[:, i, :]
        k_t = key[:, i, :]
        v_t = value[:, i, :]
        g_t = g[:, i, :].unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, i, :].unsqueeze(-1)

        last_recurrent_state = last_recurrent_state * g_t
        v_mem = k_t.unsqueeze(-2) @ last_recurrent_state
        delta = (v_t - v_mem.squeeze(-2)) * beta_t
        last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        attn_out[:, i, :] = (q_t.unsqueeze(-2) @ last_recurrent_state).squeeze(-2)

    return attn_out, last_recurrent_state



class GatedDeltaNet(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_value_heads: int,
        num_key_heads: int,
        value_head_dim: int,
        key_head_dim: int,
        conv_kernel_size: int,
        layer_index: int
    ):
        super(GatedDeltaNet, self).__init__()
        
        self.layer_index = layer_index
        self.hidden_size = hidden_size
        self.num_v_heads = num_value_heads
        self.num_k_heads = num_key_heads
        self.v_head_dim = value_head_dim
        self.k_head_dim = key_head_dim
        self.conv_kernel_size = conv_kernel_size
        self.key_dim = num_key_heads * key_head_dim
        self.value_dim = num_value_heads * value_head_dim
        
        proj_size_qkvz = 2 * self.key_dim + 2 * self.value_dim
        proj_size_ba = 2 * num_value_heads
        self.proj_qkvz = nn.Linear(hidden_size, proj_size_qkvz, bias=False)
        self.proj_ba = nn.Linear(hidden_size, proj_size_ba, bias=False)


        self.conv_dim = 2 * self.key_dim + self.value_dim
        self.conv1d = nn.Conv1d(
            self.conv_dim,
            self.conv_dim,
            kernel_size=conv_kernel_size,
            padding=conv_kernel_size - 1,
            groups=self.conv_dim,
            bias=False
        )

        self.dt_bias = nn.Parameter(torch.ones(num_value_heads))

        A = torch.empty(num_value_heads).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))


        self.norm = RMSNormGated(value_head_dim, 1e-6)
        self.proj_out = nn.Linear(self.value_dim, hidden_size, bias=False)


        self.causal_conv1d_update = torch_causal_conv1d_update
        self.fix_query_key_value_ordering = fix_query_key_value_ordering
        self.recurrent_gated_delta_rule = torch_recurrent_gated_delta_rule

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: Cache | None = None,
        attention_mask: torch.Tensor | None = None,
    ):
        hidden_states = apply_mask_to_padding_state(
            hidden_states,
            attention_mask
        )

        if cache_params:
            conv_state = cache_params.layers[self.layer_index].conv_state
            recurrent_state = cache_params.layers[self.layer_index].recurrent_state

        projected_states_qkvz = self.proj_qkvz(hidden_states)
        projected_states_ba = self.proj_ba(hidden_states)
        query, key, value, z, b, a = self.fix_query_key_value_ordering(
            projected_states_qkvz,
            projected_states_ba,
            self.num_v_heads,
            self.num_k_heads,
            self.v_head_dim,
            self.k_head_dim,
        )
        query, key, value = [x.reshape(*x.size()[:2], -1) for x in (query, key, value)]
        mixed_qkv = torch.cat((query, key, value), dim=-1)
        mixed_qkv = mixed_qkv.transpose(1, 2)

        if cache_params:
            if conv_state is None:
                # deal with the first inference
                conv_state = torch.zeros(
                    *mixed_qkv.shape[:-1], self.conv_kernel_size,
                    device=mixed_qkv.device, dtype=mixed_qkv.dtype
                )

            cache_params.update_conv_state(
                self.layer_index,
                torch.cat(
                    (conv_state, mixed_qkv),
                    dim=-1
                )[..., -self.conv_kernel_size:]
            )

            mixed_qkv = self.causal_conv1d_update(
                mixed_qkv,
                conv_state,
                self.conv1d.weight
            )
        else:
            # training mode
            mixed_qkv = F.silu(self.conv1d(mixed_qkv)[..., :mixed_qkv.shape[-1]])

        mixed_qkv = mixed_qkv.transpose(1, 2)
        query, key, value = torch.split(
            mixed_qkv,
            [self.key_dim, self.key_dim, self.value_dim],
            dim=-1
        )

        query = query.reshape(*query.size()[:2], -1, self.k_head_dim)
        key = key.reshape(*key.size()[:2], -1, self.k_head_dim)
        value = value.reshape(*value.size()[:2], -1, self.v_head_dim)

        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)

        attn_out, last_recurrent_state = self.recurrent_gated_delta_rule(
            query,
            key,
            value,
            g,
            beta,
            initial_recurrent_state=recurrent_state if cache_params else None,
            qk_norm=True,
        )

        if cache_params:
            cache_params.update_recurrent_state(
                self.layer_index, last_recurrent_state,
            )

        attn_out = self.norm(attn_out, z)
        attn_out = attn_out.reshape(*attn_out.size()[:2], -1)

        out = self.proj_out(attn_out)

        return out
