#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2026/10/28 10:56
# @Author  : XinYi Huang
# @FileName: cast.py
import math

import torch
from torch import nn
from torch.nn import functional as F

from contextlib import contextmanager
from pydantic import BaseModel, Field

from typing import Optional, List

# lora
class LoRALinear(nn.Linear):
    def __init__(
        self,
        r: int = 8,
        alpha: int = 16,
        adapter: str = "default",
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.lora_A = nn.ModuleDict()
        self.lora_B = nn.ModuleDict()

        self.r = {}
        self.alpha = {}
        self.scaling = {}
        self.active = adapter

        self.add_adapter(adapter, r, alpha)


    def add_adapter(
        self,
        name: str,
        r: int = 8,
        alpha: int = 16
    ):
        self.r[name] = r
        self.alpha[name] = alpha
        self.scaling[name] = alpha / r

        A = nn.Linear(self.in_features, r, bias=False)
        B = nn.Linear(r, self.out_features, bias=False)
        nn.init.kaiming_uniform_(A.weight, a=math.sqrt(5))
        nn.init.zeros_(B.weight)

        self.lora_A[name] = A
        self.lora_B[name] = B

    def delta(self, name: str):
        return (self.lora_B[name].weight @ self.lora_A[name].weight) * self.scaling[name]


    def forward(self, x):
        o = super().forward(x)
        return o + self.lora_B[self.active](self.lora_A[self.active](x)) * self.scaling[self.active]

    def set_adapter(
        self, 
        name: str,
        inference_mode: bool = False
    ):
        if inference_mode:
            for param in list(
                *self.lora_A[name].parameters(), *self.lora_B[name].parameters()
            ):
                param.requires_grad_(False)
        self.active = name


assert torch.cuda.is_available(), "CUDA GPU IS REQUIRED"
DEV, CPU = torch.device("cuda"), torch.device("cpu")
DTYPE = torch.bfloat16


class LoraConfig(BaseModel):
    adapter_name: str = Field(default="default", description="lora adapter name", type=str)
    r: int = Field(default=8, description="lora rank", type=int)
    alpha: int = Field(default=16, description="lora alpha", type=int)
    modules: list = Field(default_factory=lambda: ['linear'], min_length=1, description="modules to apply lora", type=list)
    inference_mode: bool = Field(default=False, description="whether to apply lora in inference mode", type=bool)


@torch.no_grad()
def inject_adapter(
    model: nn.Module,
    config: LoraConfig,
):
    '''
    inject lora adapter to model, default on CPU device
    '''
    for name, module in model.named_children():
        if name in config.modules and isinstance(module, nn.Linear):
            lora_module = LoRALinear(
                config.r,
                config.alpha,
                config.adapter_name,
                module.in_features,
                module.out_features,
                bias=True if module.bias is not None else False,
            )
            lora_module.weight = module.weight
            lora_module.bias = module.bias
            lora_module.to(device=CPU, dtype=DTYPE)
            setattr(model, name, lora_module)
            continue
        try:
            inject_adapter(module, config)
        except:
            pass


class OffloadManager:
    def __init__(
        self, 
        num_streams: int = 2,
    ):
        self.num_streams = num_streams
        self.streams = [torch.cuda.Stream() for _ in range(num_streams)]
        self.stream_index = 0
        self.buffers = {}

    def get_stream(self):
        # 在时序上, 分配流 >> 计算流
        # cuda流拷贝权重的前提：务必等到计算流完成消费
        stream = self.streams[self.stream_index]
        # 计算流可能正在进行消费, 等待消费完成, 再切换分配流
        # 否则可能引发权重覆写或驱逐, 引发推理异常
        stream.wait_stream(torch.cuda.current_stream())
        self.stream_index = (self.stream_index + 1) % self.num_streams

        return self.streams[self.stream_index]

    def get_buffer(
        self,
        stream: torch.cuda.Stream,
        numel: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        # 各流仅拷贝一份buffer
        buffer = self.buffers.get(stream, None)
        if buffer is None or buffer.numel() < numel:
            if buffer is not None:
                torch.cuda.synchronize()
            buffer = torch.empty(numel, dtype=dtype, device=device)
            self.buffers[stream] = buffer

        return buffer[:numel]


offload_manager = OffloadManager()

@contextmanager
def cast_weight_bias(
    module: nn.Module,
    input: torch.Tensor,
):
    stream = None
    try:
            # 1.cuda常驻层
        if module.weight.device == input.device:
            weight = module.weight
            bias = module.bias
            
            if module.weight_function is not None:
                for func in module.weight_function:
                    weight = func(weight)

            args = (weight, bias)
            if isinstance(module, nn.LayerNorm):
                normalized_shape = (*module.normalized_shape,)
                args = (normalized_shape,) + args

            yield args

        else:
            # 2.lowvram 转移层
            params = [p for p in (module.weight, module.bias) if p is not None]
            total = sum(p.numel() for p in params)

            # 获取权重拷贝cuda流 & 权重buffer
            stream = offload_manager.get_stream()
            buffer = offload_manager.get_buffer(stream, total, input.dtype, input.device)

            # 确保buffer在对应流中拷贝
            with torch.cuda.stream(stream):
                offset = 0
                for p in params:
                    # non_blocking=True 异步权重流送
                    buffer[offset:offset + p.numel()].copy_(p.flatten(), non_blocking=True)
                    offset += p.numel()

            # (推理至当前层时)计算流必须等待分配流权重拷贝完成, 否则可能引发设备不一致异常
            torch.cuda.current_stream().wait_stream(stream)
            weight = buffer[:module.weight.numel()].view_as(module.weight)
            bias = buffer[-module.bias.numel():].view_as(module.bias) if module.bias is not None else None
            
            if module.weight_function is not None:
                for func in module.weight_function:
                    weight = func(weight)
            args = (weight, bias)

            if isinstance(module, nn.LayerNorm):
                normalized_shape = (*module.normalized_shape,)
                args = (normalized_shape,) + args

            yield args

    finally:
        if stream is not None:
            stream.wait_stream(torch.cuda.current_stream())


class CastMixin:
    def __init__(
        self, 
        cast_weights: bool = False,
        exec: callable = None,
        weight_function: Optional[List[callable]] = None 
    ):
        self.cast_weights = cast_weights
        self.exec = exec
        self.weight_function = weight_function

    def extra_exec_args(self):
        return ()

    def forward_cast_weights(self, *args, **kwargs):
        with cast_weight_bias(self, *args) as additional_args:
            return self.exec(*args, *additional_args, *self.extra_exec_args())


class CastLinear(nn.Linear, CastMixin):
    def __init__(self, *args, **kwargs):
        cast_weights = kwargs.pop("cast_weights", False)
        weight_function = kwargs.pop("weight_function", [])
        CastMixin.__init__(self, cast_weights=cast_weights, exec=F.linear, weight_function=weight_function)
        super().__init__(*args, **kwargs)

    def forward(self, *args, **kwargs):
        if self.cast_weights or self.weight_function is not None:
            return self.forward_cast_weights(*args, **kwargs)
        return super().forward(*args, **kwargs)

class CastConv2d(nn.Conv2d, CastMixin):
    def __init__(self, *args, **kwargs):
        cast_weights = kwargs.pop("cast_weights", False)
        weight_function = kwargs.pop("weight_function", [])
        CastMixin.__init__(self, cast_weights=cast_weights, exec=F.conv2d, weight_function=weight_function)
        super().__init__(*args, **kwargs)
            
    def forward(self, *args, **kwargs):
        if self.cast_weights or self.weight_function is not None:
            return self.forward_cast_weights(*args, **kwargs)
        return super().forward(*args, **kwargs)

    def extra_exec_args(self):
        return (
            self.stride, 
            self.padding, 
            self.dilation, 
            self.groups,
        )

class CastConv1d(nn.Conv1d, CastMixin):
    def __init__(self, *args, **kwargs):
        cast_weights = kwargs.pop("cast_weights", False)
        weight_function = kwargs.pop("weight_function", [])
        CastMixin.__init__(self, cast_weights=cast_weights, exec=F.conv1d, weight_function=weight_function)
        super().__init__(*args, **kwargs)
            
    def forward(self, *args, **kwargs):
        if self.cast_weights or self.weight_function is not None:
            return self.forward_cast_weights(*args, **kwargs)
        return super().forward(*args, **kwargs)

    def extra_exec_args(self):
        return (
            self.stride, 
            self.padding, 
            self.dilation, 
            self.groups,
        )

class CastLayerNorm(nn.LayerNorm, CastMixin):
    def __init__(self, *args, **kwargs):
        cast_weights = kwargs.pop("cast_weights", False)
        kwargs.pop("weight_function", [])
        CastMixin.__init__(self, cast_weights=cast_weights, exec=F.layer_norm)
        super().__init__(*args, **kwargs)
            
    def forward(self, *args, **kwargs):
        if self.cast_weights or self.weight_function is not None:
            return self.forward_cast_weights(*args, **kwargs)
        return super().forward(*args, **kwargs)

    def extra_exec_args(self):
        return (
            self.eps,
        )


BASE_CANDIDATE_MODULES = [
    nn.Linear,
    nn.Conv1d,
    nn.Conv2d,
    nn.LayerNorm,
    LoRALinear,
]

CAST_MODULES_MAP = {
    LoRALinear: CastLinear,
    nn.Linear: CastLinear,
    nn.Conv1d: CastConv1d,
    nn.Conv2d: CastConv2d,
    nn.LayerNorm: CastLayerNorm,
}

def castize(
    model: nn.Module,
    budget: int
):
    used: int = 0
    @torch.no_grad()
    def _castize(
        model: nn.Module,
        budget: int
    ):
        nonlocal used
        for name, module in model.named_children():
            if module.__class__ in BASE_CANDIDATE_MODULES:
                size = sum([p.numel() * p.element_size() for p in module.parameters() if p is not None])
                used += size

                if used > budget:
                    kwargs = {
                        "bias": True if module.bias is not None else False,
                        "cast_weights": True,
                        "weight_function": None,
                    }
                    if isinstance(module, (nn.Linear, LoRALinear)):
                        kwargs["in_features"] = module.in_features
                        kwargs["out_features"] = module.out_features
                        if isinstance(module, LoRALinear):
                            lora_weight = module.delta(module.active)
                            kwargs["weight_function"] = [lambda weight: weight + lora_weight.to(weight)]
                    elif isinstance(module, (nn.Conv1d, nn.Conv2d)):
                        if module.padding_mode != "zeros":
                            # non-zero padding is not supported at present
                            module.to(DEV, dtype=DTYPE)
                            continue
                        kwargs["in_channels"] = module.in_channels
                        kwargs["out_channels"] = module.out_channels
                        kwargs["kernel_size"] = module.kernel_size
                        kwargs["stride"] = module.stride
                        kwargs["padding"] = module.padding
                        kwargs["dilation"] = module.dilation
                        kwargs["groups"] = module.groups
                    elif isinstance(module, nn.LayerNorm):
                        kwargs["normalized_shape"] = module.normalized_shape
                        kwargs["elementwise_affine"] = module.elementwise_affine
                        kwargs["eps"] = module.eps

                    cast_module = CAST_MODULES_MAP[module.__class__](
                        **kwargs
                    )
                    cast_module.weight = module.weight
                    cast_module.bias = module.bias
                    cast_module.weight.data = cast_module.weight.data.pin_memory()
                    if cast_module.bias is not None:
                        cast_module.bias.data = cast_module.bias.data.pin_memory()
                    cast_module.to(CPU)
                    setattr(model, name, cast_module)
                else:
                    module.to(device=DEV, dtype=DTYPE)
            
            try:
                _castize(module, budget)
            except:
                continue

    _castize(model, budget)


if __name__ == '__main__':
    # 测试模型
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear_blocks = nn.ModuleList([nn.Linear(1024, 1024) for _ in range(12)])
            self.linear = nn.Linear(1024, 32)
            self.layer_norm = nn.LayerNorm(32)

        def forward(self, x):
            for block in self.linear_blocks:
                x = block(x)
            x = self.linear(x)
            x = self.layer_norm(x)
            return x

    m = M().to(device=CPU, dtype=DTYPE)
    # 注入Lora
    config = LoraConfig()
    inject_adapter(m, config)

    x = torch.randn(4, 128, 1024).to(device=DEV, dtype=DTYPE)

    # 计算模型参数量
    total = 0
    for param in m.parameters():
        total += param.numel() * param.element_size()
    
    # 设置预算, 开始权重流送
    castize_func = lambda model, budget: castize(model, budget)
    castize_func(m, total // 2)
    y = m(x)
