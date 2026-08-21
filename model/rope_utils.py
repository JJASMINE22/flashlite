#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2026/04/27 12:55
# @Author  : XinYi Huang
# @FileName: rope_utils.py
# @Software: Trae CN
import math

import torch
from torch.nn import functional as F

from transformers import PretrainedConfig

from typing import Optional


def _compute_default_rope_parameters(
        config: PretrainedConfig,
        device: Optional[torch.device] = None
):
    base = config.rope_theta
    partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
    head_dim = config.head_dim if hasattr(config, "head_dim") else config.hidden_size // config.num_attention_heads
    dim = int(head_dim * partial_rotary_factor)

    inv_freq = 1 / (base ** (torch.arange(0, dim, 2).to(device, torch.float) / dim))

    return inv_freq


def _compute_linear_rope_parameters(
        config: PretrainedConfig,
        device: Optional[torch.device] = None
):
    factor = config.rope_scaling["factor"]

    inv_freq = _compute_default_rope_parameters(config, device)

    return inv_freq / factor


def _compute_llama3_rope_parameters(
    config: PretrainedConfig,
    device: Optional[torch.device] = None
):
    
    inv_freq = _compute_default_rope_parameters(config, device)
    attention_factor = 1.0
    
    factor = config.rope_scaling["factor"]
    low_freq_factor = config.rope_scaling["low_freq_factor"]
    high_freq_factor = config.rope_scaling["high_freq_factor"]
    old_context_len = config.rope_scaling["original_max_position_embeddings"]

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor

    wavelen = 2 * math.pi / inv_freq
    # maintain high frequency components
    # low frequency components dividing scale factor
    inv_freq_llama = torch.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
    # define the smooth region
    smooth_factor = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
    smooth_inv_freq = (1 - smooth_factor) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
    
    is_medium = (high_freq_wavelen <= wavelen) * (wavelen <= low_freq_wavelen)
    
    inv_freq_llama = torch.where(is_medium, smooth_inv_freq, inv_freq_llama)

    return inv_freq_llama, attention_factor
    

ROPE_INIT_FUNCTIONS = {
    "linear": _compute_linear_rope_parameters,
    "default": _compute_default_rope_parameters,
    "llama3": _compute_llama3_rope_parameters
}
