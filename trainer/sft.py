"""SFT：在预训练底座上注入"工具调用 + 简单对话"。

两个要点：
1. 只对 assistant 段计 loss（数据管线已做掩码），否则模型会把容量花在背命令输出上；
2. 工具数据 : 闲聊数据 ≈ 7:3，比例太低会退化成纯聊天，太高会丢掉自然语言流畅度。

跑法：
  python -m trainer.sft --base checkpoints/base.pt --epochs 3
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from model.model import MiniGPT
from trainer.data import SFTDataset, collate
from trainer.utils import (cosine_lr, describe_env, dump_json, load_ckpt,
                           make_optimizer, save_ckpt)

ROOT = Path(__file__).resolve().parent.parent


def evaluate(model: MiniGPT, dl: DataLoader, dev: str) -> float:
    model.eval()
    total, n = 0.0, 0
    use_amp = dev == "cuda" and torch.cuda.is_bf16_supported()
    with torch.no_grad(), torch.autocast(dev, dtype=torch.bfloat16 if use_amp else torch.float32,
                                         enabled=use_amp):
        for x, y in dl:
            x, y = x.to(dev), y.to(dev)
            _, loss = model(x, y, loss_mask=(y != -100).long())
            total += loss.item()
            n += 1
    model.train()
    return total / max(1, n)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="checkpoints/base.pt")
    ap.add_argument("--data", default="data/tokenized/sft.pt")
    ap.add_argument("--out", default="checkpoints/sft.pt")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1.2e-4)
    ap.add_argument("--warmup", type=int, default=60)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--from-scratch", action="store_true", help="不加载底座（仅用于冒烟测试）")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {describe_env()}")

    ck_path = ROOT / a.base
    if ck_path.exists() and not a.from_scratch:
        model, ck, cfg = load_ckpt(ck_path)
        print(f"底座: {a.base} step={ck.get('step')} 参数={model.num_params()/1e6:.1f}M")
    else:
        from model.model import ModelConfig
        cfg = ModelConfig()
        model = MiniGPT(cfg)
        print(f"从零初始化（冒烟模式）参数={model.num_params()/1e6:.1f}M")
    model.to(dev)

    blob = torch.load(ROOT / a.data, weights_only=False)
    items = blob["items"]
    pad_id = blob["pad_id"]
    n_val = max(16, int(len(items) * 0.02))
    val_items, train_items = items[:n_val], items[n_val:]
    print(f"SFT 样本: 训练 {len(train_items)} / 验证 {n_val} | pad={pad_id}")

    train_dl = DataLoader(SFTDataset(train_items), batch_size=a.batch, shuffle=True,
                          num_workers=0, collate_fn=lambda b: collate(b, pad_id), drop_last=True)
    val_dl = DataLoader(SFTDataset(val_items), batch_size=a.batch, shuffle=False,
                        num_workers=0, collate_fn=lambda b: collate(b, pad_id))

    opt = make_optimizer(model, a.lr)
    steps_per_epoch = len(train_dl) // a.accum
    total_steps = max(1, int(steps_per_epoch * a.epochs))
    print(f"总步数 {total_steps}（每 epoch {steps_per_epoch} 步）")

    use_amp = dev == "cuda" and torch.cuda.is_bf16_supported()
    model.train()
    t0 = time.time()
    losses: list[float] = []
    step = 0
    for epoch in range(int(math.ceil(a.epochs))):
        for i, (x, y) in enumerate(train_dl):
            if step >= total_steps:
                break
            lr = cosine_lr(step, total_steps, a.warmup, a.lr, min_ratio=0.1)
            for g in opt.param_groups:
                g["lr"] = lr
            x, y = x.to(dev), y.to(dev)
            with torch.autocast(dev, dtype=torch.bfloat16 if use_amp else torch.float32, enabled=use_amp):
                _, loss = model(x, y, loss_mask=(y != -100).long())
            (loss / a.accum).backward()
            losses.append(loss.item())

            if (i + 1) % a.accum == 0:
                if a.clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1

                if step % a.log_every == 0:
                    avg = sum(losses[-a.log_every * a.accum:]) / min(len(losses), a.log_every * a.accum)
                    print(f"epoch {epoch+1} step {step}/{total_steps} | loss {avg:.4f} | "
                          f"lr {lr:.2e} | {sum(losses[-50:])/min(50,len(losses)):.4f}(ma50)", flush=True)

                if step % a.eval_every == 0:
                    vl = evaluate(model, val_dl, dev)
                    print(f"  [验证] step {step} loss {vl:.4f} ppl {math.exp(min(20, vl)):.1f}", flush=True)

    vl = evaluate(model, val_dl, dev)
    save_ckpt(ROOT / a.out, model, opt, step, cfg, val_loss=vl, stage="sft")
    print(f"\nSFT 完成：{step} 步，验证 loss {vl:.4f}，耗时 {(time.time()-t0)/60:.1f} 分钟")
    print(f"已保存 {a.out}")
    dump_json(ROOT / "out/sft_summary.json", {
        "steps": step, "val_loss": vl, "train_loss_ma50": sum(losses[-50:]) / min(50, len(losses)),
        "elapsed_sec": time.time() - t0, "samples": len(train_items),
    })


if __name__ == "__main__":
    main()
