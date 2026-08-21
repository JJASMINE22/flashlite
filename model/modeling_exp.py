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


if __name__ == "__main__":
    from transformers import DeepseekV3ForCausalLM