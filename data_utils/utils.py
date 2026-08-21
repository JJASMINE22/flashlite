#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2026/05/11 14:33
# @Author  : XinYi Huang
# @FileName: utils.py
# @Software: Trae CN
import json
import torch
import random

from typing import Any, List


def replace_keep_case(match):
    word = match.group(0)
    if word.lower() == "minimind":
        if word.isupper():
            return "FLASHLITE"
        elif word[0].isupper():
            return "FlashLite"
        return "flashlite"
    elif word.lower() == "jingyaogong":
        if word.isupper():
            return "IGOR"
        elif word[0].isupper():
            return "Igor"
        return "igor"
    return word
    


def pre_processing_chat(conversations: List):
    if conversations[0].get("role") == "system":
        if conversations[0].get("content") or eval(conversations[0].get("tools")):
            return conversations
        else:
            conversations = conversations[1:]
    return [{"role": "system", "content": "你是一个专业的AI助手，请提供有价值的回答。"}] + conversations


def post_processing_chat(prompt: str, empty_think_ratio: float = .2):
    if "<think>\n\n</think>\n\n" in prompt and random.random() >= empty_think_ratio:
        prompt = prompt.replace("<think>\n\n</think>\n\n", "")
    return prompt


def check_valid_agent_conversation(conversations: List):
    if not conversations:
        raise ValueError("Conversations is empty")
        
    for i, messege in enumerate(conversations):
        role = messege.get("role")

        if role == "assistant":
            content = messege.get("content", "").strip()
            tool_calls = messege.get("tool_calls",[])
            
            if not content and not tool_calls:
                raise ValueError("Assistant content is empty")

            if i + 1 < len(conversations):
                next_role = conversations[i+1].get("role")
                if next_role == "tool" and not tool_calls:
                    continue


def create_chat_prompt(
    conversations: List,
    tokenizer: Any
):
    
    tools = conversations[0].get("tools", None)
    if tools: tools = json.loads(tools) if isinstance(tools, str) else tools

    messages = []
    for message in conversations:
        if "minimind" in message.get("content"):
            message["content"] = message["content"].replace("minimind", "flashlite")
        if "jingyaogong" in message.get("content"):
            message["content"] = message["content"].replace("jingyaogong", "igor")
        if message.get("tool_calls") and isinstance(message["tool_calls"], str):
            message["tool_calls"] = json.loads(message["tool_calls"])

        messages.append(message)

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False, 
        tools=tools
    )


def generate_labels(
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        bos_token_id: torch.Tensor
    ):
        
        unfold_input_ids = input_ids.unfold(1, len(bos_token_id), 1)
        _, st_ids = unfold_input_ids.eq(bos_token_id).all(-1).nonzero(as_tuple=True)
        
        assert st_ids.numel() > 0, "No valid assistant content"
        if st_ids.numel() == 1:
            assert st_ids[0] + len(bos_token_id) < attention_mask.sum(), "No valid assistant content"
        
        st_ed_ids = torch.cat(
                [
                    st_ids,
                    torch.tensor([attention_mask.sum()], dtype=st_ids.dtype)
                ]
            )
        st_ed_chunks = st_ed_ids.unfold(0, 2, 1)

        labels = torch.full(input_ids.size(), -100, dtype=torch.long)
        for st, ed in st_ed_chunks:
            labels[:, st + len(bos_token_id):ed] = input_ids[:, st + len(bos_token_id):ed]
        
        return labels