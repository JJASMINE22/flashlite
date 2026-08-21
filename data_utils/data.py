#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2026/04/27 11:28
# @Author  : XinYi Huang
# @FileName: data.py
# @Software: Trae CN
import re
import json
import random
from pathlib import Path

import pandas as pd

import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info
import torch.distributed as dist

from transformers import AutoTokenizer

import sys
sys.path.append("./")

from data_utils.utils import (
    replace_keep_case,
    pre_processing_chat, 
    post_processing_chat, 
    check_valid_agent_conversation
)

from typing import Any, Tuple, Optional, List, Dict


class PretrainedDataset(Dataset):
    def __init__(
        self,
        args: Any,
        task: str,
        tokenizer: AutoTokenizer
    ):
        super().__init__()
        self.args = args
        
        data_path = list(Path(args.data_dir).glob("*.jsonl"))[0]
        data = pd.read_json(
            data_path,
            lines=True
        )
        data = data.sample(frac=1, random_state=args.seed)
        if task == "train":
            self.data = data.iloc[:int(len(data) * args.train_ratio)]
        else:
            self.data = data.iloc[int(len(data) * args.train_ratio):]
            
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index) -> Optional[Tuple]:
        try:
            text, = self.data.iloc[index].values
            encoding = self.tokenizer(
                text,
                padding="max_length",
                max_length=self.args.max_length - 1,
                truncation=True
            )
            input_ids = [self.tokenizer.bos_token_id] + encoding.input_ids
            attn_mask = [1] + encoding.attention_mask
            input_ids = torch.tensor(input_ids, dtype=torch.long)
            attn_mask = torch.tensor(attn_mask, dtype=torch.float)
            labels = torch.where(
                attn_mask == 1,
                input_ids,
                -100
            )

            return input_ids, attn_mask, labels
            
        except Exception as e:
            return


class StreamingPretrainedDataset(IterableDataset):
    def __init__(
        self,
        args: Any,
        tokenizer: AutoTokenizer
    ):
        super().__init__()
        self.args = args
        self.train_ratio = args.train_ratio

        data_paths = list(Path(args.data_dir).glob("*.jsonl"))
        self.data_paths = data_paths
            
        self.tokenizer = tokenizer
        self.pad_token_id = self.tokenizer.pad_token_id
        self.eot_token_id = self.pad_token_id
        self.max_length = self.args.max_length

    def _get_shard_info(self):
        if dist.is_initialized():
            world_size = dist.get_world_size()
            rank = dist.get_rank()
        else:
            world_size = 1
            rank = 0

        worker_info = get_worker_info()
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        total_shards = world_size * num_workers
        global_shard_id = rank * num_workers + worker_id
        
        return global_shard_id, total_shards

    def _file_generator(
        self,
        file_path: str,
        global_shard_id: int,
        total_shards: int,
        buffer_size: int = 10000
    ):
        buffer = []
        while True:
            with open(file_path, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    # align the unique line to the global shard id
                    if i % total_shards != global_shard_id:
                        continue

                    data = json.loads(line)

                    buffer.append(data)
                    if len(buffer) >= buffer_size:
                        random_idx = random.randint(0, len(buffer) - 1)
                        yield buffer.pop(random_idx)
            
            random.shuffle(buffer)
            while buffer:
                yield buffer.pop()

        
    def __iter__(self) -> Optional[Tuple]:
        # get present shard info
        global_shard_id, total_shards = self._get_shard_info()
        file_generators = {
            file_path: self._file_generator(file_path, global_shard_id, total_shards)
            for file_path in self.data_paths
        }
        
        while True:
            chosen_file = random.choice(self.data_paths)
            try:
                data = next(file_generators[chosen_file])
                encoding = self.tokenizer(
                        data["text"]
                    )
                input_ids = encoding.input_ids + [self.eot_token_id]
                attention_mask = encoding.attention_mask + [1]

                input_ids = input_ids[:self.max_length]
                attention_mask = attention_mask[:self.max_length]
                
                labels = input_ids + [-100] * (self.max_length - len(input_ids))
                input_ids = input_ids + [self.pad_token_id] * (self.max_length - len(input_ids))
                attention_mask = attention_mask + [0] * (self.max_length - len(attention_mask))
                
                yield (
                    torch.tensor(input_ids, dtype=torch.long),
                    torch.tensor(attention_mask, dtype=torch.float),
                    torch.tensor(labels, dtype=torch.long)
                )
                
            except Exception as e:
                continue


class StreamingSFTDataset(IterableDataset):
    def __init__(
        self,
        args: Any,
        tokenizer: AutoTokenizer
    ):
        super().__init__()
        self.args = args
        self.max_length = args.max_length

        data_paths = list(Path(args.data_dir).glob("*.jsonl"))
        self.data_paths = data_paths
            
        self.tokenizer = tokenizer

        bot_st_token = tokenizer.bos_token + "assistant\n"
        self.bot_st_token_id = tokenizer(bot_st_token, add_special_tokens=False, return_tensors="pt").input_ids[0]

        bot_ed_token = tokenizer.bos_token + "user\n"
        self.bot_ed_token_id = tokenizer(bot_ed_token, add_special_tokens=False, return_tensors="pt").input_ids[0]

    def _get_shard_info(self):
        if dist.is_initialized():
            world_size = dist.get_world_size()
            rank = dist.get_rank()
        else:
            world_size = 1
            rank = 0

        worker_info = get_worker_info()
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        total_shards = world_size * num_workers
        global_shard_id = rank * num_workers + worker_id
        
        return global_shard_id, total_shards

    def _file_generator(
        self,
        file_path: str,
        global_shard_id: int,
        total_shards: int,
        buffer_size: int = 10000
    ):
        buffer = []
        while True:
            with open(file_path, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    # align the unique line to the global shard id
                    if i % total_shards != global_shard_id:
                        continue

                    data = json.loads(line)

                    buffer.append(data)
                    if len(buffer) >= buffer_size:
                        random_idx = random.randint(0, len(buffer) - 1)
                        yield buffer.pop(random_idx)
            
            random.shuffle(buffer)
            while buffer:
                yield buffer.pop()

    def create_chat_prompt(
        self,
        conversations: List
    ):

        def deduplicate_conversations(conversations):
            if not conversations:
                return conversations
            convs = [conversations[0]]
            for conversation in conversations[1:]:
                if convs[-1]["role"] == conversation["role"]:
                    continue
                else:
                    convs.append(conversation)
            return convs

        conversations = deduplicate_conversations(conversations)

        if not conversations:
            raise ValueError("Invalid conversation format")

        tools = conversations[0].get("tools", None)
        if tools: tools = json.loads(tools) if isinstance(tools, str) else tools

        messages = []
        for message in conversations:
            content = message.get("content")
            reasoning_content = message.get("reasoning_content")
            
            if content:
                message["content"] = re.sub(r"minimind|jingyaogong", replace_keep_case, content, flags=re.IGNORECASE)
            if reasoning_content:
                message["reasoning_content"] = re.sub(r"minimind|jingyaogong", replace_keep_case, reasoning_content, flags=re.IGNORECASE)
            if message.get("tool_calls") and isinstance(message["tool_calls"], str):
                message["tool_calls"] = json.loads(message["tool_calls"])

            messages.append(message)

        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False, 
            tools=tools
        )

    def generate_labels(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ):
        
        unfold_st_input_ids = input_ids.unfold(1, len(self.bot_st_token_id), 1)
        _, st_ids = unfold_st_input_ids.eq(self.bot_st_token_id).all(-1).nonzero(as_tuple=True)

        unfold_ed_input_ids = input_ids.unfold(1, len(self.bot_ed_token_id), 1)
        _, ed_ids = unfold_ed_input_ids.eq(self.bot_ed_token_id).all(-1).nonzero(as_tuple=True)
        
        assert st_ids.numel() > 0, "No valid assistant content"
        if st_ids.numel() == 1:
            assert st_ids[0] + len(self.bot_st_token_id) < attention_mask.sum(), "No valid assistant content"

        ed_ids = torch.cat(
            [
                ed_ids[1:],
                torch.tensor([attention_mask.sum()], dtype=st_ids.dtype)
            ]
        )

        assert ed_ids.numel() >= st_ids.numel(), "Invalid conversation format" 

        labels = torch.full(input_ids.size(), -100, dtype=torch.long)
        for st, ed in zip(st_ids, ed_ids):
            labels[:, st + len(self.bot_st_token_id):ed] = input_ids[:, st + len(self.bot_st_token_id):ed]
        
        return labels
        
    def __iter__(self) -> Optional[Tuple]:
        # get present shard info
        global_shard_id, total_shards = self._get_shard_info()
        file_generators = {
            file_path: self._file_generator(file_path, global_shard_id, total_shards)
            for file_path in self.data_paths
        }
        
        while True:
            chosen_file = random.choice(self.data_paths)
            try:
                data = next(file_generators[chosen_file])
                conversations = pre_processing_chat(data["conversations"])
                check_valid_agent_conversation(conversations)
                
                prompt = self.create_chat_prompt(conversations)
                prompt = post_processing_chat(prompt)

                encodings = self.tokenizer(
                    prompt,
                    padding="max_length",
                    max_length=self.args.max_length,
                    truncation=True,
                    return_tensors="pt"
                )
                
                labels = self.generate_labels(
                    encodings.input_ids,
                    encodings.attention_mask
                )

                yield (
                    encodings.input_ids.squeeze(0),
                    encodings.attention_mask.squeeze(0),
                    labels.squeeze(0)
                )
                
            except Exception as e:
                continue


class DPODataset(Dataset):
    def __init__(
        self,
        args: Any,
        tokenizer: AutoTokenizer
    ):
        super().__init__()
        self.args = args
        
        data_path = list(Path(args.data_dir).glob("*.jsonl"))[0]
        data = pd.read_json(
            data_path,
            lines=True
        )
        data = data.sample(frac=1, random_state=args.seed)
        self.data = data
            
        self.tokenizer = tokenizer
        
        bot_st_token = tokenizer.bos_token + "assistant\n"
        self.bot_st_token_id = tokenizer(bot_st_token, add_special_tokens=False, return_tensors="pt").input_ids[0]

        bot_ed_token = tokenizer.bos_token + "user\n"
        self.bot_ed_token_id = tokenizer(bot_ed_token, add_special_tokens=False, return_tensors="pt").input_ids[0]

    def __len__(self):
        return len(self.data)

    def create_chat_prompt(
        self,
        conversations: List
    ):

        def deduplicate_conversations(conversations):
            if not conversations:
                return conversations
            convs = [conversations[0]]
            for conversation in conversations[1:]:
                if convs[-1]["role"] == conversation["role"]:
                    continue
                else:
                    convs.append(conversation)
            return convs

        conversations = deduplicate_conversations(conversations)

        if not conversations:
            raise ValueError("Invalid conversation format")

        tools = conversations[0].get("tools", None)
        if tools: tools = json.loads(tools) if isinstance(tools, str) else tools

        messages = []
        for message in conversations:
            content = message.get("content")
            reasoning_content = message.get("reasoning_content")
            
            if content:
                message["content"] = re.sub(r"minimind|jingyaogong", replace_keep_case, content, flags=re.IGNORECASE)
            if reasoning_content:
                message["reasoning_content"] = re.sub(r"minimind|jingyaogong", replace_keep_case, reasoning_content, flags=re.IGNORECASE)
            if message.get("tool_calls") and isinstance(message["tool_calls"], str):
                message["tool_calls"] = json.loads(message["tool_calls"])

            messages.append(message)

        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False, 
            tools=tools
        )


    def generate_mask(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ):
        
        unfold_st_input_ids = input_ids.unfold(1, len(self.bot_st_token_id), 1)
        _, st_ids = unfold_st_input_ids.eq(self.bot_st_token_id).all(-1).nonzero(as_tuple=True)

        unfold_ed_input_ids = input_ids.unfold(1, len(self.bot_ed_token_id), 1)
        _, ed_ids = unfold_ed_input_ids.eq(self.bot_ed_token_id).all(-1).nonzero(as_tuple=True)
        
        assert st_ids.numel() > 0, "No valid assistant content"
        if st_ids.numel() == 1:
            assert st_ids[0] + len(self.bot_st_token_id) < attention_mask.sum(), "No valid assistant content"

        ed_ids = torch.cat(
            [
                ed_ids[1:],
                torch.tensor([attention_mask.sum()], dtype=st_ids.dtype)
            ]
        )

        assert ed_ids.numel() >= st_ids.numel(), "Invalid conversation format" 

        mask = torch.zeros_like(input_ids, dtype=torch.float)
        for st, ed in zip(st_ids, ed_ids):
            mask[:, st + len(self.bos_token_id):ed].data.fill_(1.0)
        
        return mask
    
    def __getitem__(self, index) -> Optional[Tuple]:
        try:
            chosen_chats, rejected_chats, = self.data.iloc[index].values
            
            # process chosen chats
            chosen_chats = pre_processing_chat(chosen_chats)
            check_valid_agent_conversation(chosen_chats)
            
            chosen_prompt = self.create_chat_prompt(chosen_chats)
            chosen_prompt = post_processing_chat(chosen_prompt)
            chosen_encodings = self.tokenizer(
                chosen_prompt,
                padding="max_length",
                max_length=self.args.max_length,
                truncation=True,
                return_tensors="pt"
            )

            chosen_mask = self.generate_mask(
                chosen_encodings.input_ids,
                chosen_encodings.attention_mask
            )
            
            # process rejected chats
            rejected_chats = pre_processing_chat(rejected_chats)
            check_valid_agent_conversation(rejected_chats)

            rejected_prompt = self.create_chat_prompt(rejected_chats)
            rejected_prompt = post_processing_chat(rejected_prompt)
            rejected_encodings = self.tokenizer(
                rejected_prompt,
                padding="max_length",
                max_length=self.args.max_length,
                truncation=True,
                return_tensors="pt"
            )
            
            rejected_mask = self.generate_mask(
                rejected_encodings.input_ids,
                rejected_encodings.attention_mask
            )

            return (
                chosen_encodings.input_ids.squeeze(0),
                chosen_encodings.attention_mask.squeeze(0),
                chosen_mask.squeeze(0),
                rejected_encodings.input_ids.squeeze(0),
                rejected_encodings.attention_mask.squeeze(0),
                rejected_mask.squeeze(0)
            )
            
            
        except Exception as e:
            return


class RLAIFDataset(IterableDataset):
    def __init__(
        self,
        args: Any,
        tokenizer: AutoTokenizer
    ):
        super().__init__()
        self.args = args
        self.tokenizer = tokenizer
        
        data_path = list(Path(args.data_dir).glob("*.jsonl"))[0]
        data = pd.read_json(
            data_path,
            lines=True
        )
        data = data.sample(frac=1, random_state=args.seed).reset_index(drop=True)
        self.data = data

    def _get_shard_info(self):
        if dist.is_initialized():
            world_size = dist.get_world_size()
            rank = dist.get_rank()
        else:
            world_size = 1
            rank = 0

        worker_info = get_worker_info()
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        total_shards = world_size * num_workers
        global_shard_id = rank * num_workers + worker_id
        
        return global_shard_id, total_shards

    def create_chat_prompt(
        self,
        conversations: List
    ):
        def deduplicate_conversations(conversations):
            if not conversations:
                return conversations
            deduplicated_convs = [conversations[0]]
            for conversation in conversations[1:]:
                if deduplicated_convs[-1]["role"] == conversation["role"]:
                    continue
                else:
                    deduplicated_convs.append(conversation)
            return deduplicated_convs
        
        def reduce_conversations(
            messages: List,
            tools: Optional[Dict] = None
        ):
            system_message = messages.pop(0)
            while len(messages) > 1:
                curr_messages = [system_message] + messages

                temp_prompt = self.tokenizer.apply_chat_template(
                    curr_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    tools=tools
                )

                token_count = len(self.tokenizer.encode(temp_prompt, add_special_tokens=False))

                if token_count <= self.args.max_prompt_length:
                    break

                if len(messages) > 1:
                    messages = messages[2:]
                else:
                    break
            
            messages = [system_message] + messages

            return messages


        conversations = deduplicate_conversations(conversations)

        while conversations and conversations[-1].get("role") != "user":
            conversations.pop()

        if len(conversations) < 2:
            raise ValueError("Invalid conversation format")

        check_valid_agent_conversation(conversations)

        tools = conversations[0].get("tools", None)
        if tools: tools = json.loads(tools) if isinstance(tools, str) else tools

        messages = []
        for message in conversations:
            content = message.get("content")
            reasoning_content = message.get("reasoning_content")
            
            if content:
                message["content"] = re.sub(r"minimind|jingyaogong", replace_keep_case, content, flags=re.IGNORECASE)
            if reasoning_content:
                message["reasoning_content"] = re.sub(r"minimind|jingyaogong", replace_keep_case, reasoning_content, flags=re.IGNORECASE)
            
            if message.get("tool_calls") and isinstance(message["tool_calls"], str):
                message["tool_calls"] = json.loads(message["tool_calls"])

            messages.append(message)

        messages = reduce_conversations(messages, tools)

        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            tools=tools
        ), tools

    def _build_item(self, idx):
        """
        由单行数据构造一个样本, 失败抛异常由调用方重试。
        把"无效样本过滤"收敛到迭代器内部, 保证 batch 永不出现 None。
        """
        args = self.data.iloc[idx].values.tolist()
        conversations = args.pop(0)
        ground_truths = args[0] if args else []

        conversations = pre_processing_chat(conversations)
        prompt_text, tools = self.create_chat_prompt(conversations)
        prompt_text = post_processing_chat(prompt_text, 0.)

        return {
            "prompt_texts": prompt_text,
            "tools": tools if tools else [],
            "ground_truths": ground_truths
        }

    def __iter__(self):
        global_shard_id, total_shards = self._get_shard_info()
        owned_indices = list(
            range(global_shard_id, len(self.data), total_shards)
        )

        while True:
            random.shuffle(owned_indices)
            for idx in owned_indices:
                try:
                    yield self._build_item(idx)
                except Exception:
                    continue


def collate_fn_packed(batch_data):
    input_ids = torch.tensor([item["input_ids"] for item in batch_data], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids, dtype=torch.float)

    labels = input_ids.clone()
    # Attenrion: 0 -> <|endoftext|>
    ignore_mask = labels.eq(0)
    ignore_mask = ignore_mask.roll(shifts=1, dims=1)

    labels[ignore_mask] = -100

    return (
        input_ids,
        attention_mask,
        labels
    )

def grpo_collate_fn(batch_data) -> Optional[Dict]:

    batch_data = [data for data in batch_data if data is not None]
    if not batch_data:
        return

    return {
        "prompt_texts": [data["prompt_texts"] for data in batch_data],
        "tools": [data["tools"] for data in batch_data],
        "ground_truths": [data["ground_truths"] for data in batch_data]
    }
            

if __name__ == "__main__":
    from pathlib import Path
    # import sys 
    # sys.path.append("./")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        "./tokenizer"
    )
    
    from configuration import DataArgs
    from base_utils.parse_utils import CustomArgumentParser
    from torch.utils.data import DataLoader
    
    parser = CustomArgumentParser(DataArgs)

    data_args, = parser.parse_args_into_dataclasses()

    data_dir = Path(data_args.data_dir)
    data_paths = list(data_dir.glob("*.jsonl"))
    data_args.data_paths = data_paths

    dataset = StreamingSFTDataset(data_args, tokenizer)
    # dataset = RLAIFDataset(data_args, tokenizer)
    # dataset = StreamingPretrainedDataset(data_args, tokenizer)

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        num_workers=0,
        # collate_fn=grpo_collate_fn
    )
    
    from tqdm import tqdm

    for batch_data in tqdm(dataloader):
        input_ids, attn_masks, _ = batch_data
        # prompt_texts, tools, ground_truths = batch_data.values()
        # for prompt, tool, ground_truth in zip(prompt_texts, tools, ground_truths):
        #     print(prompt)
        #     print("==" * 50)
        for input_id, attn_mask in zip(input_ids, attn_masks):
            print(tokenizer.decode(input_id, skip_special_tokens=True))
            print("==" * 50)
            1
