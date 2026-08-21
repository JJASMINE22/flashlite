#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2026/03/09 11:42
# @Author  : XinYi Huang
# @FileName: utils.py
# @Software: PyCharm
import os
import json
import logging

import torch


class HParams:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if type(v) == dict:
                v = HParams(**v)
            setattr(self, k, v)

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    # def __setitem__(self, key, value):
    #     return setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()

    def get(self, index):
        return self.__dict__.get(index)


class InferHParams(HParams):
    def __init__(self, **kwargs):
        super(InferHParams, self).__init__(**kwargs)

    def __getattr__(self, index):
        return self.get(index)


def get_hparams_from_file(config_path, infer_mode=False):
    with open(config_path, "r") as f:
        data = f.read()
    config = json.loads(data)
    hparams = HParams(**config) if not infer_mode else InferHParams(**config)
    return hparams


def latest_checkpoint_path(model_dir: str):
    files = os.listdir(model_dir)
    files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))

    return os.path.join(model_dir, files[-1])


def clean_checkpoint(model_dir: str, n_ckpt_to_keep: int = 5):
    files = os.listdir(model_dir)
    files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))

    if files.__len__() > n_ckpt_to_keep:
        for file in files[:-n_ckpt_to_keep]:
            os.remove(os.path.join(model_dir, file))


def load_checkpoint(ckptpath, model, optimizer=None):
    ckpt_dict = torch.load(ckptpath, map_location="cpu")

    if optimizer is not None:
        try:
            optimizer.load_state_dict(ckpt_dict["optimizer"])
        except KeyError as e:
            logging.info(repr(e))

    if hasattr(model, "module"):
        model.module.load_state_dict(ckpt_dict["weight"], strict=False)
    else:
        model.load_state_dict(ckpt_dict["weight"], strict=False)

    logging.info("{} loaded successfully".format(ckptpath))

    try:
        step = ckpt_dict["step"]
    except KeyError as e:
        logging.info(repr(e))
        step = 0

    return step


def load_pretrained(
        model_cls,
        pretrained_dir,
        attn_implementation=None,
        torch_dtype=None,
        strict=False
):
    """
    安全加载 HF 格式模型, 绕过 transformers 5.x from_pretrained 对自定义
    PreTrainedModel 子类的权重 materialization bug(参见 issue #46620 / #43646):
    save 的权重正确, 但 from_pretrained 会 silently 返回随机权重 / 用 _init_weights 覆盖训练值。

    改用: config.from_pretrained(只读 config.json, 无 materialization) + 实例化 + load_state_dict,
    与 load_checkpoint 等价, 已验证正确。
    """
    config = model_cls.config_class.from_pretrained(pretrained_dir)
    if attn_implementation is not None:
        config._attn_implementation = attn_implementation

    model = model_cls(config)
    if torch_dtype is not None:
        model = model.to(torch_dtype)

    state_dict = _load_state_dict_from_dir(pretrained_dir)

    model.load_state_dict(state_dict, strict=strict)
    return model


def _load_state_dict_from_dir(pretrained_dir):
    """
    从 HF 格式目录读取 state_dict, 支持四种落盘形式:
      - model.safetensors.index.json + model-XXXX-of-YYYY.safetensors (分片 safetensors)
      - model.safetensors (单文件 safetensors)
      - pytorch_model.bin.index.json + pytorch_model-XXXX-of-YYYY.bin (分片 bin)
      - pytorch_model.bin (单文件 bin)
    """
    from safetensors.torch import load_file as safe_load_file

    # 1) 分片 safetensors
    idx = os.path.join(pretrained_dir, "model.safetensors.index.json")
    if os.path.exists(idx):
        with open(idx, "r") as f:
            weight_map = json.load(f)["weight_map"]
        state_dict = {}
        for shard_name in sorted(set(weight_map.values())):
            state_dict.update(safe_load_file(os.path.join(pretrained_dir, shard_name)))
        return state_dict

    # 2) 单文件 safetensors
    st = os.path.join(pretrained_dir, "model.safetensors")
    if os.path.exists(st):
        return safe_load_file(st)

    # 3) 分片 bin
    idx_b = os.path.join(pretrained_dir, "pytorch_model.bin.index.json")
    if os.path.exists(idx_b):
        with open(idx_b, "r") as f:
            weight_map = json.load(f)["weight_map"]
        state_dict = {}
        for shard_name in sorted(set(weight_map.values())):
            state_dict.update(torch.load(os.path.join(pretrained_dir, shard_name), map_location="cpu"))
        return state_dict

    # 4) 单文件 bin
    bn = os.path.join(pretrained_dir, "pytorch_model.bin")
    if os.path.exists(bn):
        return torch.load(bn, map_location="cpu")

    raise FileNotFoundError(
        f"no weights (model.safetensors / sharded *.safetensors / pytorch_model.bin) found in {pretrained_dir}"
    )


def save_checkpoint(model, optimizer, step, ckptpath):
    if hasattr(model, "module"):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    torch.save(
        {'weight': state_dict,
         'optimizer': None if optimizer is None else optimizer.state_dict(),
         'step': step}, ckptpath
    )

from enum import Enum
from dataclasses import dataclass
from typing import Optional


class Stage(Enum):
    s1 = "TrainingStage1Args"
    s2 = "TrainingStage2Args"


@dataclass
class Trainingstate:
    stage: Optional[Stage] = None

    def set_stage(self, stage: Stage):
        self.stage = stage

    def clear_stage(self):
        self.stage = None

    def active(self, stage: Stage):
        return self.stage == stage

    def __str__(self):
        """string prompt"""
        status = {stage: self.stage == stage for stage in Stage}
        return f"TrainingStage({', '.join(f'{k.value}={v}' for k, v in status.items())})"
