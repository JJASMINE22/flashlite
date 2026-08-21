#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2026/04/27 13:04
# @Author  : XinYi Huang
# @FileName: train_pretrain_packed.py
# @Software: Trae CN
import os
import sys

sys.path.append("./")

import logging
from tqdm import tqdm
import multiprocessing

import torch
import torch.distributed as dist
from torch import multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler

from trainer.trainer_utils import (
    FlashLiteForCausalLM,
    TrainerUtils,
    AutoTokenizer
)

from datasets import load_from_disk
from data_utils.data import collate_fn_packed
from base_utils.utils import clean_checkpoint, save_checkpoint
from base_utils.parse_utils import CustomArgumentParser

from configuration import (
    FlashLiteConfig,                       
    PretrainingArgs,
    DataArgs
)

from logger.log_config import log_config

log_config("pretraining_packed")


def main():
    assert torch.cuda.is_available()

    parser = CustomArgumentParser(
        [
            PretrainingArgs,
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
    
    model = FlashLiteForCausalLM(FlashLiteConfig()).cuda(rank)
    tokenizer = AutoTokenizer.from_pretrained(data_args.tokenizer_dir)

    dataset = load_from_disk(data_args.packed_data_dir)
    sampler = DistributedSampler(dataset, shuffle=True)
    # when data preprocessing involves GPU inference, may throw exceptions
    num_workers = 4 if multiprocessing.cpu_count() > 4 else multiprocessing.cpu_count()

    data_loader = DataLoader(
        dataset, num_workers=num_workers, shuffle=False, pin_memory=True,
        batch_size=training_args.batch_size, sampler=sampler,
        collate_fn=collate_fn_packed
    )
    
    total_steps = training_args.epoches * len(data_loader)
    trainer = TrainerUtils(
        training_args,
        model,
        tokenizer,
        rank,
        total_steps
    )

    for epoch in range(training_args.epoches):
        sampler.set_epoch(epoch)
        # training part
        for _, items in enumerate(tqdm(data_loader, desc=f"train steps",
                                    total=data_loader.__len__())):
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
                if rank == 0:
                    save_checkpoint(
                        trainer.model, None, trainer.step + 1,
                        os.path.join(training_args.ckpt_dir,
                                     "trainer_{}.pth.tar".format(trainer.step + 1))
                    )
                
                    clean_checkpoint(training_args.ckpt_dir, training_args.keep_ckpts)

                trainer.train_loss = 0.
                trainer.train_aux_loss = 0.


if __name__ == '__main__':
    main()
