#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2026/04/07 14:17
# @Author  : XinYi Huang
# @FileName: train_sft.py
# @Software: Trae CN
import os
import sys

sys.path.append("./")

import wandb
import logging
from tqdm import tqdm
import multiprocessing

import torch
import torch.distributed as dist
from torch import multiprocessing as mp
from torch.utils.data import DataLoader

from trainer.trainer_utils import (
    FlashLiteForCausalLM,
    TrainerUtils,
    AutoTokenizer
)

from datasets import load_dataset
from data_utils.data import StreamingSFTDataset

from data_utils.utils import (
    pre_processing_chat,
    post_processing_chat,
    create_chat_prompt,
    generate_labels
)

from base_utils.utils import clean_checkpoint, save_checkpoint
from base_utils.parse_utils import CustomArgumentParser

from configuration import (
    FlashLiteConfig,                       
    SFTArgs,
    DataArgs
)

from logger.log_config import log_config

log_config("sft")


def log_metrics(rank, n_gpus, trainer, log_steps):
    loss = torch.tensor([trainer.train_loss]).cuda(rank)
    aux_loss = torch.tensor([trainer.train_aux_loss]).cuda(rank)

    dist.all_reduce(loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(aux_loss, op=dist.ReduceOp.SUM)

    if rank == 0:
        avg_loss = loss.item() / (n_gpus * log_steps)
        avg_aux_loss = aux_loss.item() / (n_gpus * log_steps)
        
        wandb.log(
            {
                "train/loss": avg_loss,
                "train/aux_loss": avg_aux_loss,
                "train/lr": trainer.optimizer.param_groups[0]["lr"],
                "train/step": trainer.step + 1
            }
        )


def main():
    assert torch.cuda.is_available()

    parser = CustomArgumentParser(
        [
            SFTArgs,
            DataArgs
        ]
    )

    (
        training_args,
        data_args
    ) = parser.parse_args_into_dataclasses()

    n_gpus = torch.cuda.device_count()
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = training_args.port

    mp.spawn(
        run,
        nprocs=n_gpus,
        args=(n_gpus, training_args, data_args)
    )


def run(rank, n_gpus, *args):
    (
        training_args,
        data_args
    ) = args

    # for pytorch on win, backend use gloo
    dist.init_process_group(
        backend='gloo' if os.name == 'nt' else 'nccl', init_method='env://?use_libuv=False',
        world_size=n_gpus, rank=rank
    )
    torch.cuda.set_device(rank)

    if rank == 0:
        # init wandb
        wandb.init(
            entity="cache0708",
            project="FlashLite-SFT",
            config={
                "architecture": "CausalLM"
            },
        )
    
    model = FlashLiteForCausalLM(FlashLiteConfig()).cuda(rank)
    tokenizer = AutoTokenizer.from_pretrained(data_args.tokenizer_dir)
    
    # Using Customized Dataset
    train_dataset = StreamingSFTDataset(data_args, tokenizer)

    # when data preprocessing involves GPU inference, may throw exceptions
    num_workers = 4 if multiprocessing.cpu_count() > 4 else multiprocessing.cpu_count()

    train_loader = DataLoader(
        train_dataset, num_workers=num_workers, shuffle=False, pin_memory=True,
        batch_size=training_args.batch_size
    )
    
    trainer = TrainerUtils(
        training_args,
        model,
        tokenizer,
        rank,
        training_args.total_steps
    )

    # training part
    for _, items in enumerate(tqdm(train_loader, desc=f"train steps",
                                   total=training_args.total_steps)):
        trainer.train(items)

        if rank == 0:
            if not (trainer.step + 1) % training_args.gen_steps:
                trainer.sample(items)

        if not (trainer.step + 1) % training_args.log_steps:
            lr = trainer.optimizer.param_groups[0]["lr"]
            logging.info(
                f'gpu: {rank}\n'
                f'steps: {trainer.step + 1}\n'
                f'lr: {lr}\n'
                f'train loss is: {trainer.train_loss / training_args.log_steps}\n'
                f'train aux loss is: {trainer.train_aux_loss / training_args.log_steps}\n'
            )

            log_metrics(rank, n_gpus, trainer, training_args.log_steps)

            if rank == 0:
                save_checkpoint(
                    trainer.model, None, trainer.step + 1,
                    os.path.join(training_args.ckpt_dir,
                                 "trainer_{}.pth.tar".format(trainer.step + 1))
                )
            
                clean_checkpoint(training_args.ckpt_dir, training_args.keep_ckpts)

            trainer.train_loss = 0.
            trainer.train_aux_loss = 0.

    if rank == 0:
        wandb.finish()


def make_process_fn(tokenizer, max_length):
    def process_fn(data):
        conversations = pre_processing_chat(data["conversations"])
        prompt = create_chat_prompt(conversations, tokenizer)
        prompt = post_processing_chat(prompt)

        encodings = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt"
        )

        bos_token = tokenizer.bos_token + "assistant\n"
        bos_token_id = tokenizer(bos_token, add_special_tokens=False, return_tensors="pt").input_ids[0]

        labels = generate_labels(
            encodings.input_ids,
            encodings.attention_mask,
            bos_token_id
        )

        return {
                "input_ids": encodings.input_ids.squeeze(0),
                "attention_mask": encodings.attention_mask.squeeze(0),
                "labels": labels.squeeze(0)
            }

    return process_fn


def run_hf_ds(rank, n_gpus, *args):
    (
        training_args,
        data_args
    ) = args

    # for pytorch on win, backend use gloo
    dist.init_process_group(
        backend='gloo' if os.name == 'nt' else 'nccl', init_method='env://?use_libuv=False',
        world_size=n_gpus, rank=rank
    )
    torch.cuda.set_device(rank)

    if rank == 0:
        # init wandb
        wandb.init(
            entity="cache0708",
            project="FlashLite-SFT",
            config={
                "architecture": "CausalLM"
            },
        )
    
    model = FlashLiteForCausalLM(FlashLiteConfig()).cuda(rank)
    tokenizer = AutoTokenizer.from_pretrained(data_args.tokenizer_dir)

    dataset = load_dataset(
        "json", 
        data_dir=data_args.data_dir,
        streaming=True,
        split="train"
    )
    sharded_dataset = dataset.shard(n_gpus, rank)

    columns=list(next(iter(sharded_dataset)).keys())
    process_fn = make_process_fn(tokenizer, data_args.max_length)
    processed_dataset = sharded_dataset.map(process_fn, remove_columns=columns)

    trainer = TrainerUtils(
        training_args,
        model,
        tokenizer,
        rank,
        training_args.total_steps
    )

    # when data preprocessing involves GPU inference, may throw exceptions
    num_workers = 4 if multiprocessing.cpu_count() > 4 else multiprocessing.cpu_count()

    for epoch in range(training_args.epoches):
        train_dataset = processed_dataset.shuffle(seed=data_args.seed + epoch)
        train_loader = DataLoader(
            train_dataset, num_workers=num_workers, shuffle=False, pin_memory=True,
            batch_size=training_args.batch_size
        )

        for _, items in enumerate(tqdm(train_loader, desc="train steps")):
            trainer.train(items)

            if rank == 0:
                if not (trainer.step + 1) % training_args.gen_steps:
                    trainer.sample(items)

            if not (trainer.step + 1) % training_args.log_steps:
                lr = trainer.optimizer.param_groups[0]["lr"]
                logging.info(
                    f'gpu: {rank}\n'
                    f'steps: {trainer.step + 1}\n'
                    f'lr: {lr}\n'
                    f'train loss is: {trainer.train_loss / training_args.log_steps}\n'
                    f'train aux loss is: {trainer.train_aux_loss / training_args.log_steps}\n'
                )

                log_metrics(rank, n_gpus, trainer, training_args.log_steps)

                if rank == 0:
                    save_checkpoint(
                        trainer.model, None, trainer.step + 1,
                        os.path.join(training_args.ckpt_dir,
                                     "trainer_{}.pth.tar".format(trainer.step + 1))
                    )
                
                    clean_checkpoint(training_args.ckpt_dir, training_args.keep_ckpts)

                trainer.train_loss = 0.
                trainer.train_aux_loss = 0.

    if rank == 0:
        wandb.finish()
    


if __name__ == '__main__':
    main()
