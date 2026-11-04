"""训练公共件：学习率、保存/加载、耗时估算。"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch

from model.model import MiniGPT, ModelConfig


def cosine_lr(step: int, total: int, warmup: int, base_lr: float, min_ratio: float = 0.05) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    progress = min(1.0, progress)
    return base_lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))


def make_optimizer(model: MiniGPT, lr: float, weight_decay: float = 0.1) -> torch.optim.Optimizer:
    """norm 与 embedding 不做 weight decay。"""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or "embed" in name:
            no_decay.append(p)
        else:
            decay.append(p)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95), eps=1e-8)


def save_ckpt(path: str | Path, model: MiniGPT, optimizer, step: int, cfg: ModelConfig, **extra) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer else None,
        "step": step,
        "config": cfg.__dict__,
        **extra,
    }, path)


def load_ckpt(path: str | Path, map_location="cpu", with_optim: bool = False):
    ck = torch.load(path, map_location=map_location, weights_only=False)
    cfg = ModelConfig(**ck["config"])
    model = MiniGPT(cfg)
    model.load_state_dict(ck["model"])
    return model, ck, cfg


def fmt_time(sec: float) -> str:
    sec = int(sec)
    h, m = divmod(sec // 60, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{sec % 60:02d}s"


class Meter:
    def __init__(self, total: int):
        self.t0 = time.time()
        self.total = total
        self.step = 0

    def update(self, step: int) -> tuple[float, str]:
        self.step = step
        el = time.time() - self.t0
        rate = el / max(1, step)
        return el, fmt_time(rate * max(0, self.total - step))


def describe_env() -> str:
    if not torch.cuda.is_available():
        return "CPU（训练会非常慢）"
    p = torch.cuda.get_device_properties(0)
    return f"{p.name} {p.total_memory/1024**3:.1f}GB sm{p.multi_processor_count} bf16={torch.cuda.is_bf16_supported()}"


def dump_json(path: str | Path, obj: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
