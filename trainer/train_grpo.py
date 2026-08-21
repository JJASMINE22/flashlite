#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2026/04/30
# @Author  : XinYi Huang
# @FileName: train_grpo.py
# @Software: Trae CN
import os
import sys

sys.path.append("./")

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

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
    RLAIFTrainerUtils,
    AutoTokenizer
)

from data_utils.data import RLAIFDataset, grpo_collate_fn

from base_utils.utils import clean_checkpoint, save_checkpoint, load_pretrained
from base_utils.parse_utils import CustomArgumentParser

from configuration import (
    GRPOArgs,
    DataArgs
)

from logger.log_config import log_config

log_config("grpo")


def log_metrics(rank, n_gpus, trainer, log_steps):
    metrics = torch.tensor([
        trainer.train_loss,
        trainer.train_aux_loss,
        trainer.train_reward,
        trainer.train_advantage
    ]).cuda(rank)

    dist.all_reduce(metrics, op=dist.ReduceOp.SUM)

    monitor = torch.tensor([
        float(trainer.train_group_sim),
        float(trainer.train_reward_std),
        float(trainer.train_low_signal),
        float(trainer.train_total_groups)
    ]).cuda(rank)
    dist.all_reduce(monitor, op=dist.ReduceOp.SUM)

    if rank == 0:
        avg_loss, avg_aux_loss, avg_reward, avg_advantage = (
            (metrics / (n_gpus * log_steps)).tolist()
        )

        m_sim, m_std, m_low, m_tot = monitor.tolist()
        avg_group_sim = m_sim / (n_gpus * log_steps)
        avg_reward_std = m_std / (n_gpus * log_steps)
        low_signal_ratio = (m_low / m_tot) if m_tot > 0 else 0.0

        wandb.log(
            {
                "train/policy_loss": avg_loss,
                "train/aux_loss": avg_aux_loss,
                "train/reward": avg_reward,
                "train/advantage": avg_advantage,
                "train/lr": trainer.optimizer.param_groups[0]["lr"],
                "train/step": trainer.step + 1,
                "train/group_similarity": avg_group_sim,
                "train/reward_std": avg_reward_std,
                "train/low_signal_ratio": low_signal_ratio
            }
        )


def main():
    assert torch.cuda.is_available()

    parser = CustomArgumentParser(
        [
            GRPOArgs,
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

    dist.init_process_group(
        backend='gloo' if os.name == 'nt' else 'nccl', init_method='env://?use_libuv=False',
        world_size=n_gpus, rank=rank
    )
    torch.cuda.set_device(rank)

    if rank == 0:
        # init wandb
        wandb.init(
            entity="cache0708",
            project="FlashLite-GRPO",
            config={
                "architecture": "CausalLM"
            },
        )

    # 绕过 transformers 5.x from_pretrained 对自定义模型的 materialization bug(#46620/#43646),
    # 改用 config.from_pretrained + load_state_dict(与 load_checkpoint 等价, 权重正确)
    model = load_pretrained(
        FlashLiteForCausalLM,
        training_args.pretrained_dir,
        attn_implementation="flash_attention_2"
    ).cuda(rank)
    tokenizer = AutoTokenizer.from_pretrained(data_args.tokenizer_dir)
    
    train_dataset = RLAIFDataset(
        data_args, tokenizer
    )

    num_workers = 4 if multiprocessing.cpu_count() > 4 else multiprocessing.cpu_count()

    train_loader = DataLoader(
        train_dataset,
        num_workers=num_workers,
        pin_memory=True,
        batch_size=training_args.batch_size,
        collate_fn=grpo_collate_fn
    )

    total_steps = training_args.total_steps

    trainer = RLAIFTrainerUtils(
        training_args=training_args,
        model=model,
        tokenizer=tokenizer,
        rank=rank,
        steps=total_steps
    )

    pbar = tqdm(total=total_steps, desc="grpo", disable=(rank != 0))
    micro_step = 0
    for items in train_loader:
        trainer.train(items)
        micro_step += 1
        # 累积步完成并更新后, 清理 rollout/generate 产生的碎片化显存
        if trainer.step % training_args.accumulation_steps == 0:
            torch.cuda.empty_cache()
        if rank == 0:
            pbar.update(1)

        if rank == 0:
            if not (trainer.step + 1) % training_args.gen_steps:
                trainer.sample(items)

        if not (trainer.step + 1) % training_args.log_steps:
            lr = trainer.optimizer.param_groups[0]["lr"]
            logging.info(
                f'gpu: {rank}\n'
                f'steps: {trainer.step + 1}\n'
                f'lr: {lr}\n'
                f'policy loss: {trainer.train_loss / training_args.log_steps}\n'
                f'aux loss: {trainer.train_aux_loss / training_args.log_steps}\n'
                f'mean reward: {trainer.train_reward / training_args.log_steps}\n'
                f'mean advantage: {trainer.train_advantage / training_args.log_steps}\n'
            )

            log_metrics(rank, n_gpus, trainer, training_args.log_steps)

            if rank == 0:
                save_checkpoint(
                    trainer.model, None, trainer.step + 1,
                    os.path.join(training_args.ckpt_dir,
                                 "grpo_trainer_{}.pth.tar".format(trainer.step + 1))
                )
            
                clean_checkpoint(training_args.ckpt_dir, training_args.keep_ckpts)

            trainer.train_loss = 0.
            trainer.train_aux_loss = 0.
            trainer.train_reward = 0.
            trainer.train_advantage = 0.
            trainer.train_group_sim = 0.
            trainer.train_reward_std = 0.
            trainer.train_low_signal = 0
            trainer.train_total_groups = 0

        if micro_step >= total_steps:
            break

    pbar.close()

    if rank == 0:
        wandb.finish()


if __name__ == '__main__':
    main()