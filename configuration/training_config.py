#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2026/03/12 08:33
# @Author  : XinYi Huang
# @FileName: training_config.py
# @Software: PyCharm
from dataclasses import dataclass, field


@dataclass
class PretrainingArgs:
    epoches: int = field(default=2, metadata={"help": "number of epoches"})
    total_steps: int = field(default=500000, metadata={"help": "number of steps"})
    batch_size: int = field(default=4, metadata={"help": "batch size"})
    warmup_steps: int = field(default=4000, metadata={"help": "warmup steps"})
    lr: float = field(default=4e-4, metadata={"help": "learning rate"})
    min_lr: float = field(default=1e-5, metadata={"help": "min learning rate"})
    betas: tuple = field(default=(0.9, 0.95), metadata={"help": "betas of adam"})
    epsilon: float = field(default=1e-5, metadata={"help": "epsilon of adam"})
    weight_decay: float = field(default=0.1, metadata={"help": "weight decay"})
    max_norm: float = field(default=1.0, metadata={"help": "max norm of the gradients"})
    accumulation_steps: int = field(
        default=24, metadata={"help": "number of accumulation steps"}
    )
    generation: dict = field(
        default_factory=lambda: {
            "do_sample": True,
            "top_k": 50,
            "top_p": 0.95,
            "temperature": 0.9,
            "scattered_prompt_length": 384,
            "max_new_length": 768,
            "repetition_penalty": 1.1
        },
        metadata={"help": "generation config"}
    )
    port: str = field(
        default="8001", metadata={"help": "port of the distributed training"}
    )
    log_steps: int = field(default=100, metadata={"help": "logging steps"})
    gen_steps: int = field(default=100, metadata={"help": "generating steps"})
    ckpt_dir: str = field(
        default="./checkpoint", metadata={"help": "ckpt dir to save dalle"}
    )
    resume_train: bool = field(default=False, metadata={"help": "if resume training"})
    keep_ckpts: int = field(default=5, metadata={"help": "keep checkpoints"})


@dataclass
class SFTArgs(PretrainingArgs):
    total_steps: int = field(default=2000000, metadata={"help": "number of steps"})
    warmup_steps: int = field(default=2000, metadata={"help": "warmup steps"})
    lr: float = field(default=1e-5, metadata={"help": "learning rate"})
    min_lr: float = field(default=1e-6, metadata={"help": "min learning rate"})
    resume_train: bool = field(default=True, metadata={"help": "if resume training"})


@dataclass
class GRPOArgs(PretrainingArgs):
    total_steps: int = field(default=15000, metadata={"help": "number of steps"})
    batch_size: int = field(default=1, metadata={"help": "batch size"})
    warmup_steps: int = field(default=100, metadata={"help": "warmup steps"})
    lr: float = field(default=2e-6, metadata={"help": "learning rate"})
    min_lr: float = field(default=1e-7, metadata={"help": "min learning rate"})
    accumulation_steps: int = field(
        default=8, metadata={"help": "number of accumulation steps"}
    )
    pretrained_dir: str = field(
        default="./sft", metadata={"help": "sft pretrained dir"}
    )
    reward_model_dir: str = field(
        default="./reward_model", metadata={"help": "reward model dir"}
    )
    resume_train: bool = field(default=False, metadata={"help": "if resume training"})
    num_generations: int = field(
        default=4, metadata={"help": "number of rollouts per prompt"}
    )
    max_prompt_length: int = field(
        default=384, metadata={"help": "max response length for generation"}
    )
    max_response_length: int = field(
        default=384, metadata={"help": "max response length for generation"}
    )
    rollout_chunk_size: int = field(
        default=4, metadata={"help": "chunk size for rollout generation"}
    )
    router_aux_loss_coef: float = field(
        default=1e-3, metadata={"help": "coefficient of router auxiliary loss"}
    )
    grpo_config: dict = field(
        default_factory=lambda: {
            "beta": .1,
            "clip_eps": .2
        },
        metadata={"help": "grpo configuration"}
    )
    resume_train: bool = field(default=True, metadata={"help": "if resume training"})


@dataclass
class DPOArgs(PretrainingArgs):
    epoches: int = field(default=2, metadata={"help": "number of epoches"})
    batch_size: int = field(default=2, metadata={"help": "batch size"})
    warmup_steps: int = field(default=2000, metadata={"help": "warmup steps"})
    lr: float = field(default=4e-8, metadata={"help": "learning rate"})
    min_lr: float = field(default=1e-9, metadata={"help": "min learning rate"})
    accumulation_steps: int = field(
        default=48, metadata={"help": "number of accumulation steps"}
    )
    pretrained_dir: str = field(
        default="./sft", metadata={"help": "sft pretrained dir"}
       )
    resume_train: bool = field(default=True, metadata={"help": "if resume training"})