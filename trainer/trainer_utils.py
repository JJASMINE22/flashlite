#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2026/04/27 13:01
# @Author  : XinYi Huang
# @FileName: trainer_utils.py
# @Software: Trae CN
import re
import json
from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from model.modeling import FlashLiteForCausalLM
from model.loss_utils import dpo_loss

from base_utils.utils import latest_checkpoint_path, load_checkpoint

from typing import Any, Dict, List


def mean_pairwise_sim(texts: List[str], n: int = 3) -> float:
    """组内文本两两字符 n-gram Jaccard 的均值, 用于监控 mode collapse。"""
    sets = []
    for s in texts:
        if len(s) >= n:
            sets.append(set(s[i:i + n] for i in range(len(s) - n + 1)))
        elif s:
            sets.append({s})
        else:
            sets.append(set())
    sims = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            if sets[i] and sets[j]:
                sims.append(len(sets[i] & sets[j]) / len(sets[i] | sets[j]))
    return sum(sims) / len(sims) if sims else 0.0


class TrainerUtils:
    def __init__(
        self,
        training_args: Any,
        model: FlashLiteForCausalLM,
        tokenizer: AutoTokenizer,
        rank: int,
        steps: int
    ):
        self.training_args = training_args
        self.tokenizer = tokenizer
        self.model = model

        self.rank = rank
        self.step = 0

        if training_args.resume_train:
            try:
                step = load_checkpoint(
                    latest_checkpoint_path(training_args.ckpt_dir),
                    self.model
                )

            except Exception as e:
                raise Exception(f"received incorrect params, {repr(e)}")

            self.step = step

        if self.model.vocab_size != len(tokenizer):
            self.model.resize_token_embeddings(len(tokenizer))

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=training_args.lr,
            betas=training_args.betas,
            weight_decay=training_args.weight_decay,
            eps=training_args.epsilon,
            fused=True
        )

        self.model = DDP(self.model, device_ids=[rank], find_unused_parameters=True)
        self.scaler = GradScaler(enabled=True)
        
        warmup_scheduler = LinearLR(
            self.optimizer, 
            start_factor=0.1, 
            end_factor=1.0, 
            total_iters=training_args.warmup_steps
        )
    
        decay_steps = max(1, steps // self.training_args.accumulation_steps - training_args.warmup_steps)
        decay_scheduler = CosineAnnealingLR(
            self.optimizer, 
            T_max=decay_steps, 
            eta_min=training_args.min_lr
        )

        self.schedular = SequentialLR(
            self.optimizer, 
            schedulers=[warmup_scheduler, decay_scheduler], 
            milestones=[training_args.warmup_steps]
        )

        self.train_loss, self.train_aux_loss = 0., 0.
        self.train_reward, self.train_advantage = 0., 0.
        self.valid_loss, self.valid_aux_loss = 0., 0.

        # D: 组级行为监控累计器(用于早发现 mode collapse)
        self.train_group_sim = 0.
        self.train_reward_std = 0.
        self.train_low_signal = 0
        self.train_total_groups = 0

    def train(self, batch):
        if isinstance(batch, dict):
            input_ids, attn_mask, labels = batch["input_ids"], batch["attn_mask"], batch["labels"]
        else:
            input_ids, attn_mask, labels = batch

        input_ids = input_ids.cuda(self.rank).contiguous()
        attn_mask = attn_mask.cuda(self.rank).contiguous()
        labels = labels.cuda(self.rank).contiguous()
        
        with autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attn_mask,
                labels=labels
            )

        loss = outputs.loss
        aux_loss = outputs.aux_loss
        total_loss = (loss + aux_loss) / self.training_args.accumulation_steps
        if (self.step + 1) % self.training_args.accumulation_steps == 0:
            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optimizer)
            if self.training_args.max_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.training_args.max_norm)
        
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.schedular.step()
            self.optimizer.zero_grad()
        else:
            with self.model.no_sync():
                self.scaler.scale(total_loss).backward()

        self.train_loss += loss.item()
        self.train_aux_loss += aux_loss.item()
        
        self.step += 1

    @torch.no_grad()
    def validate(self, batch):
        if isinstance(batch, dict):
            input_ids, attn_mask, labels = batch["input_ids"], batch["attn_mask"], batch["labels"]
        else:
            input_ids, attn_mask, labels = batch

        input_ids = input_ids.cuda(self.rank).contiguous()
        attn_mask = attn_mask.cuda(self.rank).contiguous()
        labels = labels.cuda(self.rank).contiguous()
        
        with autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attn_mask,
                labels=labels
            )

        loss = outputs.loss
        aux_loss = outputs.aux_loss

        self.valid_loss += loss.item()
        self.valid_aux_loss += aux_loss.item()

    def _find_last_assistant_pos(self, input_ids: torch.Tensor) -> int:
        """
        定位 input_ids 中最后一个 <bos>assistant\\n 的结束位置。
        Returns:
            int: prompt 长度（最后一个 assistant token 的结束位置）。
        Raises:
            ValueError: 未找到 assistant token。
        """
        bot_st_token = self.tokenizer.bos_token + "assistant\n"
        bot_st_token_id = self.tokenizer(
            bot_st_token, add_special_tokens=False, return_tensors="pt"
        ).input_ids[0]
        bot_st_token_id = bot_st_token_id.to(input_ids.device)

        unfold_input_ids = input_ids.unfold(1, len(bot_st_token_id), 1)
        match_mask = unfold_input_ids.eq(bot_st_token_id).all(-1)  # (B, L-w+1)

        matches = match_mask.nonzero()  # (N, 2): [batch_idx, pos_idx]

        if matches.numel() == 0:
            raise ValueError(
                f"No '{bot_st_token.strip()}' token found in input_ids. "
                f"Check that the input follows the chat template format."
            )

        # 第 2 列是位置索引，取全局最后一个匹配
        last_pos_idx = matches[:, 1].max().item()
        prompt_len = last_pos_idx + len(bot_st_token_id)

        return prompt_len
    
    @torch.no_grad()
    def sample(self, batch, num_samples: int = 1):
        if isinstance(batch, dict):
            input_ids, attn_mask = batch["input_ids"], batch["attn_mask"]
        else:
            input_ids, attn_mask, _ = batch

        max_new_tokens = self.training_args.generation["max_new_length"]
        top_p = self.training_args.generation["top_p"]
        top_k = self.training_args.generation["top_k"]
        temperature = self.training_args.generation["temperature"]
        repetition_penalty = self.training_args.generation["repetition_penalty"]

        batch_size = input_ids.shape[0]
        indices = torch.randperm(batch_size, device=input_ids.device)[:num_samples]
        input_ids = input_ids[indices].cuda(self.rank).contiguous()
        attn_mask = attn_mask[indices].cuda(self.rank).contiguous()
        
        try:
            prompt_len = self._find_last_assistant_pos(input_ids)
        except ValueError:
            return

        original_state = self.model.training
        original_router_logits = self.model.module.config.output_router_logits

        # 显存监控(诊断 generate 是否泄漏): allocated 涨=真泄漏, reserved 涨=碎片
        torch.cuda.synchronize()
        alloc_before = torch.cuda.memory_allocated() / 1e9
        reserved_before = torch.cuda.memory_reserved() / 1e9

        self.model.eval()
        self.model.module.config.output_router_logits = False

        try:
            with autocast(device_type="cuda", dtype=torch.bfloat16):
                generated_ids = self.model.module.generate(
                    input_ids=input_ids[:, :prompt_len],
                    attention_mask=attn_mask[:, :prompt_len],
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    top_p=top_p,
                    top_k=top_k,
                    temperature=temperature,
                    repetition_penalty=repetition_penalty
                )
        finally:
            self.model.train(original_state)
            self.model.module.config.output_router_logits = original_router_logits

        generated_texts = [
            self.tokenizer.decode(gid, skip_special_tokens=False)
            for gid in generated_ids
        ]

        # 显式释放 generate 产生的 GPU 张量(KV cache / generated_ids),
        # 并归还 caching allocator 的 reserved 显存, 避免多次 sample 后碎片累积 OOM
        del generated_ids
        torch.cuda.empty_cache()

        torch.cuda.synchronize()
        alloc_after = torch.cuda.memory_allocated() / 1e9
        reserved_after = torch.cuda.memory_reserved() / 1e9
        print(f"[mem] sample: allocated {alloc_before:.2f}->{alloc_after:.2f} GB, "
              f"reserved {reserved_before:.2f}->{reserved_after:.2f} GB")

        for text in generated_texts:
            print(text)


class DPOTrainerUtils(TrainerUtils):
    def __init__(
        self,
        **kwargs 
    ):
        rank = kwargs["rank"]
        model = kwargs["model"]
        tokenizer = kwargs["tokenizer"]

        if model.vocab_size != len(tokenizer):
            model.resize_token_embeddings(len(tokenizer))

        self.ref_model = deepcopy(model)

        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad_(False)
        self.ref_model = self.ref_model.cuda(rank)

        kwargs["model"] = model
        super().__init__(**kwargs)

        
    def train(self, batch):
        (
            chosen_input_ids,
            chosen_attn_mask,
            chosen_mask, 
            rejected_input_ids, 
            rejected_attn_mask,
            rejected_mask
        ) = batch
        
        chosen_input_ids = chosen_input_ids.cuda(self.rank).contiguous()
        chosen_attn_mask = chosen_attn_mask.cuda(self.rank).contiguous()
        chosen_mask = chosen_mask.cuda(self.rank).contiguous()
        rejected_input_ids = rejected_input_ids.cuda(self.rank).contiguous()
        rejected_attn_mask = rejected_attn_mask.cuda(self.rank).contiguous()
        rejected_mask = rejected_mask.cuda(self.rank).contiguous()
        
        input_ids = torch.cat(
            (chosen_input_ids, rejected_input_ids), dim=0
        )
        attn_mask = torch.cat(
            (chosen_attn_mask, rejected_attn_mask), dim=0
        )
        mask = torch.cat(
            (chosen_mask, rejected_mask), dim=0
        )
        
        with autocast(device_type="cuda", dtype=torch.bfloat16):
            with torch.no_grad():
                ref_outputs = self.ref_model(
                    input_ids=input_ids[:, :-1],
                    attention_mask=attn_mask[:, :-1]
                )
            
            
            policy_outputs = self.model(
                input_ids=input_ids[:, :-1],
                attention_mask=attn_mask[:, :-1]
            )

        loss = dpo_loss(
            ref_outputs.logits,
            policy_outputs.logits,
            input_ids[:, 1:],
            mask[:, 1:]
        )

        aux_loss = policy_outputs.aux_loss
        total_loss = (loss + aux_loss) / self.training_args.accumulation_steps
        if (self.step + 1) % self.training_args.accumulation_steps == 0:
            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optimizer)
            if self.training_args.max_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.training_args.max_norm)
        
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.schedular.step()
            self.optimizer.zero_grad()
        else:
            with self.model.no_sync():
                self.scaler.scale(total_loss).backward()

        self.train_loss += loss.item()
        self.train_aux_loss += aux_loss.item()
        
        self.step += 1

    @torch.no_grad()
    def sample(self, batch, num_samples: int = 1):
        chosen_input_ids = batch[0]
        chosen_attn_mask = batch[1]
        super().sample((chosen_input_ids, chosen_attn_mask, None), num_samples)


class RLAIFTrainerUtils(TrainerUtils):
    def __init__(self, **kwargs):
        rank = kwargs["rank"]
        model = kwargs["model"]

        model.cpu()
        self.ref_model = deepcopy(model)
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad_(False)
        model = model.cuda(rank)
        self.ref_model = self.ref_model.cuda(rank)

        kwargs["model"] = model
        super().__init__(**kwargs)

        self.reward_model = AutoModelForSequenceClassification.from_pretrained(
            self.training_args.reward_model_dir
        )
        self.reward_model = self.reward_model.cuda(rank)
        self.reward_model.eval()

        self.rm_tokenizer = AutoTokenizer.from_pretrained(
            self.training_args.reward_model_dir
        )
        if self.rm_tokenizer.pad_token is None:
            self.rm_tokenizer.pad_token = self.rm_tokenizer.eos_token

    @torch.no_grad()
    def rollout(
        self, 
        prompts_input_ids: torch.Tensor,
        prompts_attn_mask: torch.Tensor
    ):
        """
        基于策略模型对批量 prompt 进行 ``num_generations`` 次自回归采样。

        Args:
            prompts_input_ids (torch.Tensor): prompt token 序列, shape ``(B, L_p)``。
            prompts_attn_mask (torch.Tensor): 对应 attention mask, shape ``(B, L_p)``。

        Returns:
            tuple: ``(prompt_len, output_ids, full_mask, response_mask, responses)``, 
            其中 ``output_ids`` shape 为 ``(B * num_generations, L_p + L_r)``, 
            ``response_mask`` 屏蔽 padding, ``responses`` 为解码文本列表。
        """
        original_state = self.model.training
        original_router_logits = self.model.module.config.output_router_logits
        self.model.eval()
        self.model.module.config.output_router_logits = False

        repeated_input_ids = prompts_input_ids.repeat_interleave(
            self.training_args.num_generations, dim=0
        )
        repeated_attn_mask = prompts_attn_mask.repeat_interleave(
            self.training_args.num_generations, dim=0
        )
        total = repeated_input_ids.shape[0]
        chunk_size = getattr(
            self.training_args, "rollout_chunk_size", None
        ) or total
        chunk_size = max(1, chunk_size)

        try:
            output_chunks = []
            full_seq_len = prompts_input_ids.shape[1] + self.training_args.max_response_length
            with autocast(device_type="cuda", dtype=torch.bfloat16):
                for st in range(0, total, chunk_size):
                    ed = min(st + chunk_size, total)
                    chunk_out = self.model.module.generate(
                        input_ids=repeated_input_ids[st:ed],
                        attention_mask=repeated_attn_mask[st:ed],
                        max_new_tokens=self.training_args.max_response_length,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        top_p=self.training_args.generation.get("top_p", 0.95),
                        top_k=self.training_args.generation.get("top_k", 50),
                        temperature=self.training_args.generation.get("temperature", 0.7),
                        repetition_penalty=self.training_args.generation.get("repetition_penalty", 1.1),
                    )
                    if chunk_out.shape[1] < full_seq_len:
                        chunk_out = F.pad(
                            chunk_out,
                            (0, full_seq_len - chunk_out.shape[1]),
                            value=self.tokenizer.pad_token_id
                        )
                    output_chunks.append(chunk_out)
        finally:
            self.model.train(original_state)
            self.model.module.config.output_router_logits = original_router_logits

        prompt_len = prompts_input_ids.shape[1]
        output_ids = torch.cat(output_chunks, dim=0)

        full_mask = (output_ids != self.tokenizer.pad_token_id).float()

        response_ids = output_ids[:, prompt_len:]
        response_mask = (response_ids != self.tokenizer.pad_token_id).float()
        responses = self.tokenizer.batch_decode(
            response_ids,
            skip_special_tokens=True
        )

        return (
            prompt_len,
            output_ids, 
            full_mask,
            response_mask,
            responses
        )

    def compute_logps(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        n_keep: int,
        output_router_logits: bool = True
    ):
        """
        计算模型在输入序列最后 ``n_keep`` 个 token 上的对数概率, 
        用于 RLAIF / GRPO 中估计新旧策略的概率比。

        Args:
            model (nn.Module): 策略模型或参考模型。
            input_ids (torch.Tensor): 完整 token 序列, shape ``(B, L)``。
            attention_mask (torch.Tensor): 同形状的 attention mask。
            n_keep (int): 尾部需要计算 logp 的 token 数。
            output_router_logits (bool): 是否输出 MoE 路由 logits(仅 policy 需要 aux_loss)。

        Returns:
            tuple: ``(logps, aux_loss)``, 其中 ``logps`` shape 为 ``(B, n_keep, 1)``。
        """
        if n_keep <= 0:
            return input_ids.new_empty((input_ids.shape[1], 0), dtype=torch.float32), None
        # get the last (n_keep + 1) tokens' logits, and drop the last one
        # rearange position of logits to match the position of input_ids
        position_ids = attention_mask.cumsum(dim=-1)
        position_ids = F.pad(position_ids, (1, 0), value=0)[:, :-1]

        outputs = model(
            input_ids,
            attention_mask,
            position_ids,
            logits_to_keep=n_keep + 1,
            output_router_logits=output_router_logits
        )

        logits = outputs.logits[:, :-1]
        aux_loss = outputs.aux_loss

        return logits.log_softmax(-1).gather(-1, input_ids[:, -n_keep:].unsqueeze(-1)), aux_loss

    @torch.no_grad()
    def calculate_reward(
        self,
        prompts: List[str],
        responses: List[str],
        batched_tools: List[List[Dict[str, str]]],
        batched_gts: List[List[str]]
    ):
        """
        综合「规则奖励 + 奖励模型打分」为回复计算最终奖励。

        Args:
            prompts (List[str]): prompt 文本列表, 长度 ``B``。
            responses (List[str]): 回复文本列表, 长度 ``B * num_generations``。
            batched_tools (List[List[Dict]]): 每个 prompt 的工具定义。
            batched_gts (List[List[str] or None]): 各 prompt 的真实工具调用, 
                为空 / ``None`` 表示不需要调用工具。

        Returns:
            torch.Tensor: shape ``(B * num_generations,)`` 的最终奖励。
        """

        def rep_penalty(text, n=3, cap=0.5):
            toks = re.findall(r"\w+|[^\w\s]", text.lower())
            grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
            return min(cap, (len(grams) - len(set(grams))) * cap * 2 / len(grams)) if grams else 0.0

        def extract_tool_calls(text: str):
            matches = re.findall(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL)
            tool_calls, parse_error =[], False
            if matches:
                for match in matches:
                    match = match.strip()
                    match = match.replace(r"\'", r"\\'")
                    try:
                        tool_call = json.loads(match)
                        if isinstance(tool_call, dict) and ("name" in tool_call or "arguments" in tool_call):
                            tool_calls.append(tool_call)
                    except json.JSONDecodeError:
                        parse_error = True
            return tool_calls, parse_error

        def validate_tool_schema(
            model_tool_args: Dict | str,
            system_tool_args: Dict
        ):

            if isinstance(model_tool_args, str):
                try:
                    model_tool_args = json.loads(model_tool_args)
                except json.JSONDecodeError:
                    return False

            properties = system_tool_args.get("properties", {})
            req_arg_keys = system_tool_args.get("required", [])

            for req_key in req_arg_keys:
                if req_key not in model_tool_args:
                    return False
            
            for model_arg_key in model_tool_args.keys():
                if model_arg_key not in properties:
                    return False

            return True


        rewards = torch.zeros(len(responses)).cuda(self.rank)
        rm_chats = []

        batch_size = len(prompts)
        for i in range(batch_size):
            system_tool_maps = {
                tools["function"]["name"]: tools["function"].get("parameters", {}) for tools in batched_tools[i]
            }
            needs_tool = batched_gts[i] is not None and len(batched_gts[i]) > 0

            for j in range(self.training_args.num_generations):
                response_idx = i * self.training_args.num_generations + j
                response = responses[response_idx]
                answer = response
                prompt = prompts[i]

                think_st = response.find("<think>")
                think_ed = response.find("</think>")
                if response.count("<think>") == 1 and response.count("</think>") == 1 and think_st < think_ed:
                    rewards[response_idx] += .5
                    think_content, answer_content = response.rsplit("</think>", 1)
                    if len(think_content) < 32 or len(think_content) > 768:
                        rewards[response_idx] -= .8
                    answer = answer_content.strip()
                        
                elif response.count("<think>") or response.count("</think>"):
                    rewards[response_idx] -= .8
                    if "</think>" in response:
                        answer = response.rsplit("</think>", 1)[-1].strip()
                    else:
                        answer = response.replace("<think>", "").strip()

                rewards[response_idx] -= rep_penalty(answer)
                if len(answer) < 64 or len(answer) > 768:
                    rewards[response_idx] -= .5

                tool_calls, parse_error = extract_tool_calls(answer)

                if needs_tool:
                    if tool_calls or parse_error:
                        if parse_error:
                            rewards[response_idx] -= .8
                        else:
                            for tool_call in tool_calls:
                                model_tool_name = tool_call.get("name")
                                model_tool_args = tool_call.get("arguments", {})
                                if model_tool_name not in system_tool_maps:
                                    rewards[response_idx] -= .5
                                else:
                                    system_tool_args = system_tool_maps[model_tool_name]
                                    is_schema_valid = validate_tool_schema(model_tool_args, system_tool_args)
                                    if is_schema_valid:
                                        rewards[response_idx] += .5
                                    else:
                                        rewards[response_idx] -= .5
                            if len(tool_calls) != len(set(tool_calls)):
                                rewards[response_idx] -= .5
                    else:
                        rewards[response_idx] -= .8
                else:
                    if tool_calls or parse_error:
                        rewards[response_idx] -= .5

                rm_chats.append(prompt + response + self.rm_tokenizer.eos_token)
                
        rewards = rewards.clamp(min=-3.0, max=3.0)
        tokenized_chat = self.rm_tokenizer(
            rm_chats, 
            return_tensors="pt",
            padding=True,
        ).to(rewards.device)

        with autocast(device_type="cuda", dtype=torch.bfloat16):
            rm_scores = self.reward_model(**tokenized_chat).logits
        rm_scores = rm_scores.view(-1)

        return rewards + rm_scores

    def train(self, batch):
        # prompts has been added generation prompt
        if isinstance(batch, dict):
            prompts, batched_tools, batched_gts = batch["prompt_texts"], batch["tools"], batch["ground_truths"]
        else:
            prompts, batched_tools, batched_gts = batch

        encodings = self.tokenizer(
            prompts,
            padding=True,
            padding_side="left",
            return_tensors="pt"
        )

        input_ids = encodings.input_ids.cuda(self.rank).contiguous()
        attn_mask = encodings.attention_mask.cuda(self.rank).contiguous()

        (
            prompt_len,
            output_ids, 
            attn_mask,
            response_mask,
            responses
        ) = self.rollout(input_ids, attn_mask)

        n_keep = output_ids[:, prompt_len:].shape[1]

        with autocast(device_type="cuda", dtype=torch.bfloat16):
            with torch.no_grad():
                ref_per_token_logps, _ = self.compute_logps(
                    self.ref_model,
                    output_ids,
                    attn_mask,
                    n_keep,
                    output_router_logits=False
                )

            per_token_logps, aux_loss = self.compute_logps(
                self.model,
                output_ids,
                attn_mask,
                n_keep
            )

            old_per_token_logps = per_token_logps.detach()

        rewards = self.calculate_reward(
            prompts, responses, batched_tools, batched_gts
        )
        rewards = rewards.reshape(-1, self.training_args.num_generations)

        mean_r = rewards.mean(dim=-1, keepdim=True)
        std_r = rewards.std(dim=-1, keepdim=True, unbiased=False)
        # advantages > 0, the policy model is more relevant to positive answers
        # advantages < 0, the policy model is more relevant to negative answers
        advantages = (rewards - mean_r) / (std_r + 1e-4)

        # C: 弱信号组跳过。组内 std 过小 => 模型在该 prompt 上区分不出质量(全错/全套话),
        #    1/std 会把噪声级 reward 差放大成巨幅 advantage 并强化套话。置 0 拆除该放大器。
        #    (surrogate 项归零, KL 项仍轻微把策略拉回 ref, 符合预期)
        adv_std_threshold = self.training_args.grpo_config.get("adv_std_threshold", 0.15)
        low_signal = std_r.squeeze(-1) < adv_std_threshold
        advantages[low_signal] = 0.
        advantages = advantages.flatten()

        # D: 组级行为监控累计
        ng = self.training_args.num_generations
        self.train_reward_std += float(std_r.squeeze(-1).mean().item())
        self.train_low_signal += int(low_signal.sum().item())
        self.train_total_groups += int(low_signal.numel())
        for gi in range(rewards.shape[0]):
            self.train_group_sim += mean_pairwise_sim(responses[gi * ng:(gi + 1) * ng])

        kl_div = ref_per_token_logps - per_token_logps
        per_token_kl = kl_div.exp() - kl_div - 1
        # ratio > 1, the policy model is more confident with present token
        # ratio < 1, the policy model is less confident with present token
        ratio = (per_token_logps - old_per_token_logps).exp()

        clipped_ratio = ratio.clamp(1 - self.training_args.grpo_config["clip_eps"], 1 + self.training_args.grpo_config["clip_eps"])
        adv = advantages[:, None, None]
        per_token_loss_clipped = clipped_ratio * adv
        per_token_loss_unclipped = ratio * adv
        per_token_loss = -(
            torch.min(per_token_loss_clipped, per_token_loss_unclipped) - self.training_args.grpo_config["beta"] * per_token_kl
        )
        # weighted Average
        policy_loss = ((per_token_loss * response_mask.unsqueeze(-1)).sum(1) / response_mask.sum(1, keepdim=True).clamp(min=1)).mean()
        loss = (policy_loss + self.training_args.router_aux_loss_coef * aux_loss) / self.training_args.accumulation_steps

        if (self.step + 1) % self.training_args.accumulation_steps == 0:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            if self.training_args.max_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.training_args.max_norm)

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.schedular.step()
            self.optimizer.zero_grad()
        else:
            with self.model.no_sync():
                self.scaler.scale(loss).backward()

        self.train_loss += policy_loss.item()
        self.train_aux_loss += aux_loss.item()
        self.train_reward += rewards.mean().item()
        self.train_advantage += advantages.mean().item()

        self.step += 1

    @torch.no_grad()
    def sample(self, batch, num_samples: int = 1):
        if isinstance(batch, dict):
            prompts = batch["prompt_texts"]
        else:
            prompts = batch[0]

        encodings = self.tokenizer(
            prompts,
            padding=True,
            padding_side="left",
            return_tensors="pt"
        )
        super().sample(
            (encodings.input_ids, encodings.attention_mask, None),
            num_samples
        )