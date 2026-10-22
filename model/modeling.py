#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2026/05/15 11:42
# @Author  : XinYi Huang
# @FileName: modeling.py
# @Software: Trae CN
import math

import torch
from torch import nn
from torch.amp import autocast
from torch.nn import functional as F

from .modeling_utils import ALL_ATTENTION_FUNCTIONS, ACT2FN, apply_rotary_pos_emb
from .masking_utils import create_causal_mask
from .rope_utils import ROPE_INIT_FUNCTIONS
from configuration.modeling import FlashLiteConfig

from transformers import PreTrainedModel, PretrainedConfig
from transformers.generation import GenerationMixin
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import MoeModelOutputWithPast, MoeCausalLMOutputWithPast

from typing import Any, Optional, Tuple, Union


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        config,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        
        self.config = config
        if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
            self.rope_type = getattr(config.rope_scaling, "rope_type", "default")
        else:
            self.rope_type = "default"

        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq = self.rope_init_fn(config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = inv_freq

    @torch.no_grad()
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor
    ):
        inv_freq_expanded = self.inv_freq[None, None].expand(position_ids.shape[0], -1, -1).to(hidden_states.device)
        position_ids_expanded = position_ids[..., None].float()

        with autocast(device_type=hidden_states.device.type, enabled=False):
            freqs = position_ids_expanded @ inv_freq_expanded
            emb = torch.cat((freqs, freqs), dim=-1)
            cos, sin = emb.cos(), emb.sin()

        return cos.to(dtype=hidden_states.dtype), sin.to(dtype=hidden_states.dtype)


class RMSNorm(nn.Module):
    def __init__(self,
                 hidden_size: int,
                 rms_norm_eps: float
    ):
        super().__init__()

        hidden_size = (hidden_size,)
        
        self.rms_norm_eps = rms_norm_eps
        self.weight = nn.Parameter(torch.ones(hidden_size))


    def forward(
        self,
        hidden_states: torch.Tensor
    ):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.rms_norm_eps)

        return hidden_states.to(input_dtype) * self.weight


class Attention(nn.Module):
    
    def __init__(
        self,
        config,
        layer_idx: int
    ):
        super().__init__()
        assert config.num_attention_heads >= config.num_key_value_heads
        assert config.num_attention_heads % config.num_key_value_heads == 0
        assert config.hidden_size % (config.head_dim * config.num_key_value_heads) == 0

        self.config = config
        self.layer_idx = layer_idx

        self.head_dim = config.head_dim
        self.scaling = self.head_dim ** -.5
        self.num_kv_groups = config.num_attention_heads // config.num_key_value_heads

        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.attn_dropout = config.attn_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, self.head_dim * self.num_attention_heads, bias=config.attn_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, self.head_dim * self.num_key_value_heads, bias=config.attn_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, self.head_dim * self.num_key_value_heads, bias=config.attn_bias
        )
        self.o_proj = nn.Linear(
            self.head_dim * self.num_attention_heads, config.hidden_size, bias=config.attn_bias
        )

        self.q_norm = RMSNorm(config.head_dim, config.rms_norm_eps)
        self.k_norm = RMSNorm(config.head_dim, config.rms_norm_eps)

    def forward(
            self,
            hidden_states: torch.Tensor,
            position_embeddings: Tuple[torch.Tensor, torch.Tensor],
            attention_mask: Optional[torch.Tensor],
            past_key_values: Optional[Cache],
            cache_position: Optional[torch.Tensor],
            **kwargs
    ):
        B, N, C = hidden_states.shape
        hidden_shape = (B, N, -1, self.head_dim)

        q = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        k = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        v = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if past_key_values is not None:
            cache_kwargs = {"cos": cos, "sin": sin, "cache_position": cache_position}
            k, v = past_key_values.update(k, v, self.layer_idx, cache_kwargs)

        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        o, attn_weights = attention_interface(
            self, q, k, v, attention_mask, dropout=self.attn_dropout, scaling=self.scaling, **kwargs
        )

        o = o.reshape(B, N, -1)
        o = self.o_proj(o)

        return o, attn_weights
        

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, config.hidden_size, bias=False)

        self.act_fn = ACT2FN[config.hidden_act]

    def forward(
        self,
        hidden_states: torch.Tensor
    ):
        return self.down_proj(self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class MoeGate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob

        self.register_parameter("weight", nn.Parameter((torch.empty(self.num_experts, config.hidden_size))))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(
        self,
        hidden_states: torch.Tensor
    ):
        hidden_states = hidden_states.flatten(0, 1)
        router_logits = F.linear(hidden_states, self.weight)

        # calculate scores
        scores = F.softmax(router_logits.float(), dim=-1)
        scores = scores.to(router_logits.dtype)
        
        # topk
        routing_weights, selected_experts = torch.topk(scores, self.top_k, dim=-1)

        if self.norm_topk_prob:
            routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        
        return routing_weights, selected_experts, router_logits


class SparseMoeBlock(nn.Module):
    def __init__(
        self, 
        config
    ):
        super().__init__()
        self.num_experts = config.num_experts
        
        self.gate = MoeGate(config)
        self.experts = nn.ModuleList(
            [
                MLP(config) for _ in range(config.num_experts)
            ]
        )

    def forward(
        self,
        hidden_states: torch.Tensor
    ):
        shape = hidden_states.shape
        routing_weights, selected_experts, router_logits = self.gate(hidden_states)
        hidden_states = hidden_states.flatten(0, 1)
        final_hidden_states = torch.zeros_like(hidden_states)
        for i in range(self.num_experts):
            expert_layer = self.experts[i]
            batch_ids, expert_ids = torch.nonzero(selected_experts == i, as_tuple=True)
            if batch_ids.numel() > 0:
                current_hidden_states = hidden_states[batch_ids]
                expert_outputs = expert_layer(current_hidden_states)
                
                current_weights = routing_weights[batch_ids, expert_ids][:, None]
                weighted_outputs = expert_outputs * current_weights
                
                final_hidden_states.scatter_add_(0, batch_ids[:, None].expand(-1, expert_outputs.shape[1]), weighted_outputs)
                
        final_hidden_states = final_hidden_states.reshape(*shape)
        
        return final_hidden_states, router_logits


class DecoderLayer(nn.Module):
    def __init__(
            self,
            config,
            layer_idx: int
    ):
        super().__init__()

        self.attn = Attention(config, layer_idx)
        self.mlp = SparseMoeBlock(config)
        self.norm1 = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.norm2 = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
            self,
            hidden_states: torch.Tensor,
            position_embeddings: Tuple[torch.Tensor, torch.Tensor],
            attention_mask: Optional[torch.Tensor],
            past_key_values: Optional[Cache],
            cache_position: Optional[torch.Tensor],
            **kwargs
    ):
        residual = hidden_states.clone()
        hidden_states, _ = self.attn(
            self.norm1(hidden_states),
            position_embeddings,
            attention_mask,
            past_key_values,
            cache_position,
            **kwargs
        )

        hidden_states = residual + hidden_states

        residual = hidden_states.clone()
        hidden_states, router_logits = self.mlp(
            self.norm2(hidden_states)
        )
        hidden_states = residual + hidden_states
        
        return hidden_states, router_logits


class FlashLitePretrainedModel(PreTrainedModel):
    config_class = FlashLiteConfig
    _supports_attention_backend = True
    _supports_flash_attn = True
    _supports_sdpa = True
    
    def __init__(
        self,
        config
    ):
        super().__init__(config=config)

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class FlashLiteModel(FlashLitePretrainedModel):
    def __init__(
        self,
        config
    ):
        super().__init__(config=config)

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [
                DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)
            ]
        )

        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(config)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs
    ):
        if input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            # during training part, past kvs might be None
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            # make sure cache position is not None
            past_seen_tokens = 0 if past_key_values is None else past_key_values.get_seq_length()
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        attention_mask = create_causal_mask(
            **{
                "config": self.config,
                "input_embeds": inputs_embeds,
                "past_key_values": past_key_values,
                "attention_mask": attention_mask,
                "cache_position": cache_position
            }
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        all_router_logits = []
        for decoder_layer in self.layers[:self.config.num_hidden_layers]:
            hidden_states, router_logits = decoder_layer(
                hidden_states,
                position_embeddings,
                attention_mask,
                past_key_values=past_key_values,
                cache_position=cache_position,
                **kwargs
            )
            all_router_logits.append(router_logits)

        hidden_states = self.norm(hidden_states)

        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            router_logits=tuple(all_router_logits)
        )


def load_balancing_loss_func(
    gate_logits: Optional[Tuple[torch.Tensor]] = None,
    num_experts: Optional[int] = None,
    topk: int = 2,
    attention_mask: Optional[torch.Tensor] = None
):
    if gate_logits is None:
        return 0

    concatenated_gate_logits = torch.cat(gate_logits, dim=0)
    routing_weights = F.softmax(concatenated_gate_logits.float(), dim=-1)
    routing_weights = routing_weights.to(concatenated_gate_logits.dtype)
    
    _, selected_experts = torch.topk(routing_weights, k=topk, dim=-1)
    
    if attention_mask is None:
        selected_experts = selected_experts.flatten()
        tokens_per_expert = selected_experts.bincount(minlength=num_experts) / selected_experts.numel()
        router_prob_per_expert = routing_weights.mean(0)

    else:
        B, N = attention_mask.shape
        num_hidden_layers = concatenated_gate_logits.shape[0] // (B * N)

        expert_attention_mask = attention_mask[None].expand(num_hidden_layers, -1, -1).flatten().bool()

        tokens_per_expert = selected_experts[expert_attention_mask].flatten().bincount(minlength=num_experts) / selected_experts[expert_attention_mask].numel()
        router_prob_per_expert = routing_weights[expert_attention_mask].mean(0)

    # the nearer aux_loss equals to 1, the better balancing between experts
    overall_loss = (tokens_per_expert * router_prob_per_expert).sum()
    return overall_loss * num_experts

class FlashLiteForCausalLM(FlashLitePretrainedModel, GenerationMixin):
    # _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config):
        super().__init__(config)
        self.model = FlashLiteModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok

        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.LongTensor] = 0,
        **kwargs
    ):
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )
        
        outputs: MoeModelOutputWithPast = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_router_logits=output_router_logits,
            cache_position=cache_position,
            **kwargs
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        
        loss = None
        if labels is not None:
             loss = self.loss_function(logits, labels, self.vocab_size, **kwargs)
        
        aux_loss = None
        if output_router_logits:
            assert isinstance(outputs.router_logits, tuple), "router_logits must be a tuple when output_router_logits is True"
            aux_loss = load_balancing_loss_func(
                gate_logits=outputs.router_logits,
                num_experts=self.num_experts,
                topk=self.num_experts_per_tok,
                attention_mask=attention_mask
            )
            
            if loss is not None:
                loss += self.router_aux_loss_coef * aux_loss

        return MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits
        )
