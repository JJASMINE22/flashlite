#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2026/04/27 11:31
# @Author  : XinYi Huang
# @FileName: model_config.py
# @Software: Trae CN
from transformers import PretrainedConfig
from transformers.modeling_rope_utils import rope_config_validation


class FlashLiteConfig(PretrainedConfig):
    model_type = "flashlite"

    def __init__(
            self,
            vocab_size=12296,
            hidden_size=512,
            intermediate_size=1536,
            num_hidden_layers=8,
            num_attention_heads=8,
            num_key_value_heads=4,
            head_dim=64,
            hidden_act="silu",
            max_position_embeddings=32768,
            initializer_range=0.02,
            rms_norm_eps=1e-6,
            rope_theta=1000000.0,
            rope_scaling=None,
            attn_bias=False,
            attn_dropout=0.0,
            num_experts=4,
            num_experts_per_tok=2,
            norm_topk_prob=True,
            output_router_logits=True,
            router_aux_loss_coef=0.01,
            pad_token_id=0,
            tie_word_embeddings=False,
            **kwargs
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attn_bias = attn_bias
        self.attn_dropout = attn_dropout
        # Validate the correctness of rotary position embeddings parameters
        # BC: if there is a 'type' field, move it to 'rope_type'.
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self)

        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.norm_topk_prob = norm_topk_prob
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef
        self.pad_token_id = pad_token_id

        super().__init__(
            **kwargs,
            tie_word_embeddings=tie_word_embeddings
        )


class FlashLiteMLAConfig(PretrainedConfig):
    model_type = "flashlitemla"

    def __init__(
            self,
            vocab_size=12288,
            hidden_size=1024,
            intermediate_size=3072,
            num_hidden_layers=24,
            num_attention_heads=8,
            num_key_value_heads=4,
            q_lora_rank=128,
            qk_rope_head_dim=64,
            qk_nope_head_dim=128,
            kv_lora_rank=64,
            v_head_dim=128,
            hidden_act="silu",
            max_position_embeddings=32768,
            initializer_range=0.02,
            rms_norm_eps=1e-6,
            rope_theta=1000000.0,
            rope_scaling=None,
            attn_bias=False,
            attn_dropout=0.0,
            num_experts=4,
            num_experts_per_tok=2,
            norm_topk_prob=True,
            output_router_logits=True,
            router_aux_loss_coef=0.01,
            pad_token_id=0,
            tie_word_embeddings=True,
            **kwargs
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads

        self.q_lora_rank = q_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.kv_lora_rank = kv_lora_rank
        self.v_head_dim = v_head_dim
        self.qk_head_dim = qk_rope_head_dim + qk_nope_head_dim
        self.head_dim = qk_rope_head_dim

        self.hidden_act = hidden_act
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attn_bias = attn_bias
        self.attn_dropout = attn_dropout
        # Validate the correctness of rotary position embeddings parameters
        # BC: if there is a 'type' field, move it to 'rope_type'.
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self)

        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.norm_topk_prob = norm_topk_prob
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef
        self.pad_token_id = pad_token_id

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs
        )