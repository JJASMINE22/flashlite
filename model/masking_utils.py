#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2026/05/15 11:42
# @Author  : XinYi Huang
# @FileName: masking_utils.py
# @Software: Trae CN
import torch
from torch.nn import functional as F

from transformers import PretrainedConfig
from transformers.cache_utils import Cache
from transformers.utils.generic import GeneralInterface

from typing import Optional, Dict


def sdpa_mask(
        cache_position: torch.Tensor,
        kv_length: int,
        attention_mask: torch.Tensor,
        **kwargs
):
    """
    :param cache_position: (query_length,) indicating the current indices of the input sequence elements
    :param kv_length: number_of_seen_tokens + query_lens
    :param attention_mask: number_of_seen_tokens + query_lens
    :param attention_mask: number_of_seen_tokens + query_lens
    :return: sdpa attn mask
    """
    q_length = cache_position.shape[0]
    if kv_length == q_length:
        causal_mask = torch.tril(torch.ones(kv_length, kv_length))

    elif kv_length > q_length:
        causal_mask = torch.cat(
            [
                torch.ones(q_length, kv_length - q_length),
                torch.tril(torch.ones(q_length, q_length))
            ], 1
        )

    else:
        raise NotImplementedError("q_length < kv_length is not implemented yet.")

    causal_mask = causal_mask.to(attention_mask.device)
    mask = causal_mask[None, None] * attention_mask[:, None, None]

    return mask.bool()


def eager_mask(
        cache_position: torch.Tensor,
        kv_length: int,
        attention_mask: torch.Tensor,
        dtype: torch.dtype = torch.float32
):
    """
    :param cache_position: (query_length,) indicating the current indices of the input sequence elements
    :param kv_length: number_of_seen_tokens + query_lens
    :param attention_mask: number_of_seen_tokens + query_lens
    :param attention_mask: number_of_seen_tokens + query_lens
    :param dtype: dtype to use for the mask. By default, `torch.float32`.
    :return: eager attn mask
    """
    mask = sdpa_mask(
        cache_position=cache_position,
        kv_length=kv_length,
        attention_mask=attention_mask
    )

    min_dtype = torch.finfo(dtype).min
    mask = torch.where(mask.bool(), 0., min_dtype)

    return mask


def flash_attention_mask(
        cache_position: torch.Tensor,
        kv_length: int,
        attention_mask: torch.Tensor,
        **kwargs
):
    """
    :param cache_position: (query_length,) indicating the current indices of the input sequence elements
    :param kv_length: number_of_seen_tokens + query_lens
    :param attention_mask: number_of_seen_tokens + query_lens
    :param kwargs:
    :return: flash attn mask
    """
    attention_mask = attention_mask[:, -kv_length:]

    return attention_mask


ALL_MASK_ATTENTION_FUNCTIONS: Dict = {
        "sdpa": sdpa_mask,
        "eager": eager_mask,
        "flash_attention_2": flash_attention_mask
    }


def create_causal_mask(
        config: PretrainedConfig,
        input_embeds: torch.Tensor,
        past_key_values: Optional[Cache],
        attention_mask: torch.Tensor,
        cache_position: torch.Tensor
):
    layer_index = 0
    dtype = input_embeds.dtype

    if past_key_values is not None:
        # during inferencing part, get query + past kv length by using cache layer
        kv_length = past_key_values.get_mask_sizes(cache_position, layer_index)[0]
    else:
        kv_length = input_embeds.shape[1]

    mask_interface = ALL_MASK_ATTENTION_FUNCTIONS[config._attn_implementation]
    causal_mask = mask_interface(
        cache_position=cache_position,
        kv_length=kv_length,
        attention_mask=attention_mask,
        dtype=dtype
    )

    return causal_mask


if __name__ == '__main__':
    """from configuration_igor import IgorConfig
    from transformers.cache_utils import DynamicCache
    from transformers import AutoTokenizer

    config = IgorConfig.from_pretrained(
        ".",
        attn_implementation="eager",
        trust_remote_code=True
    )

    past_key_values = DynamicCache(
        config=config
    )

    tokenizer = AutoTokenizer.from_pretrained(
        ".",
        trust_remote_code=True
    )

    sentences = [
        "I love you.",
        "I am a robot.",
        "how's the weather in Boston?"
    ]

    input_encodings = tokenizer(
        sentences,
        padding=True,
        padding_side="left",
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt")

    input_ids = input_encodings.data["input_ids"].float()
    attention_mask = input_encodings.data["attention_mask"]

    mask = create_causal_mask(
        config=config,
        input_embeds=input_ids,
        past_key_values=past_key_values,
        attention_mask=attention_mask,
        cache_position=torch.arange(input_ids.shape[1])
    )
"""