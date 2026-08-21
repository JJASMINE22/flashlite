#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2026/04/27 11:22
# @Author  : XinYi Huang
# @FileName: doc_packing.py
# @Software: Trae CN
import sys
sys.path.append("./")

import numpy as np

import multiprocessing
from datasets import load_dataset, interleave_datasets
from transformers import AutoTokenizer

from configuration.data_config import DataArgs
from base_utils.parse_utils import CustomArgumentParser

parser = CustomArgumentParser(DataArgs)
args, = parser.parse_args_into_dataclasses()

tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
max_length = args.max_length
eot_token_id = tokenizer.pad_token_id


# Tokenize
def tokenize_function(examples):
    # examples["text"] is a list of strings
    outputs = tokenizer(examples["text"], add_special_tokens=False)
    # add endoftext token
    for input_ids in outputs["input_ids"]:
        input_ids.append(eot_token_id)
    return outputs

# Doc Packing
def group_texts(examples):
    # concatenate all columns
    concatenated_ids = sum(examples["input_ids"], [])
    
    # calculate total length
    total_length = (len(concatenated_ids) // max_length) * max_length
    
    # chunk ids
    packed_ids =[
        # 核心优化：强转为 int32 (NumPy 格式会被 Arrow 完美识别)
        np.array(concatenated_ids[i : i + max_length], dtype=np.int32)
        for i in range(0, total_length, max_length)
    ]
    return {"input_ids": packed_ids}


if __name__ == '__main__':
    # 加载原始 JSONL 数据
    # 这步很快，因为只是建立了映射，不会全读进内存
    raw_dataset = load_dataset(
        "json", 
        data_dir=args.data_dir,
        split="train",
        cache_dir=args.cache_dir
    )
    num_proc = multiprocessing.cpu_count() // 2 if multiprocessing.cpu_count() > 2 else 1

    print("开始 Tokenization...")
    tokenized_dataset = raw_dataset.map(
        tokenize_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=raw_dataset.column_names, # 删掉原始文本列，节省空间
        desc="Running tokenizer"
    )
    
    columns_to_keep = ["input_ids"] # 我们只关心 input_ids
    columns_to_remove = [col for col in tokenized_dataset.column_names if col not in columns_to_keep]
    tokenized_dataset = tokenized_dataset.remove_columns(columns_to_remove)

    print("开始 Doc Packing...")
    packed_dataset = tokenized_dataset.map(
        group_texts,
        batched=True,
        batch_size=1000,
        num_proc=num_proc,
        desc="Packing texts"
    )

    # 5. 保存到磁盘
    print(f"打包完成！共获得 {len(packed_dataset)} 条长度为 {max_length} 的训练数据。")
    packed_dataset.save_to_disk(args.packed_data_dir)
    