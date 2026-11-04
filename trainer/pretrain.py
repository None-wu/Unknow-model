"""预训练：在 40M 模型上跑 ~0.8B token。

8GB 显存下的经验参数：
  batch 16 × seq 512 × accum 4 = 每步 32768 token，
  ~24000 步正好 8 亿 token，4060 Laptop 约 20~26 小时。

跑法：
  python -m trainer.pretrain --steps 24000 --batch 16 --accum 4
中断后加 --resume 继续。
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from model.model import MiniGPT, ModelConfig
from trainer.data import PretrainDataset, collate
from trainer.utils import (cosine_lr, describe_env, dump_json, fmt_time,
                           load_ckpt, make_optimizer, save_ckpt)

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/tokenized/pretrain.bin")
    ap.add_argument("--out", default="checkpoints/base.pt")
    ap.add_argument("--arch", default="37m", choices=["37m", "60m"],
                    help="37m: dim576/8层 (默认) | 60m: dim768/8层/ffn2048 (约62M)")
    ap.add_argument("--steps", type=int, default=24000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--warmup", type=int, default=600)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {describe_env()}")

    cfg = ModelConfig(max_seq_len=max(1024, a.seq_len))
    if a.arch == "60m":
        cfg.dim, cfg.n_layers, cfg.n_heads, cfg.n_kv_heads, cfg.ffn_dim = 768, 8, 8, 2, 2048
    model = MiniGPT(cfg).to(dev)
    print(f"参数: {model.num_params()/1e6:.2f}M | 词表 {cfg.vocab_size} | 层 {cfg.n_layers} | dim {cfg.dim}")

    ds = PretrainDataset(ROOT / a.data, a.seq_len, num_samples=a.steps * a.batch * a.accum)
    print(f"token 流: {len(ds.data)/1e6:.1f}M tokens")
    if len(ds.data) < a.seq_len * 100:
        raise SystemExit("token 流太小，先跑 trainer.data --mode pretrain")

    dl = DataLoader(ds, batch_size=a.batch, shuffle=False, num_workers=0,
                    collate_fn=collate, drop_last=True)

    opt = make_optimizer(model, a.lr, a.weight_decay)
    start_step = 0
    ck_path = ROOT / a.out
    if a.resume and ck_path.exists():
        _, ck, _ = load_ckpt(ck_path)
        model.load_state_dict(ck["model"])
        if ck.get("optimizer"):
            opt.load_state_dict(ck["optimizer"])
        start_step = ck.get("step", 0)
        print(f"从 {ck_path.name} 恢复，step={start_step}")

    use_amp = dev == "cuda" and torch.cuda.is_bf16_supported()
    autocast = torch.autocast(dev, dtype=torch.bfloat16) if use_amp else torch.autocast(dev, enabled=False)

    model.train()
    it = iter(dl)
    t0 = time.time()
    tokens_per_step = a.batch * a.accum * a.seq_len
    losses: list[float] = []

    for step in range(start_step, a.steps):
        lr = cosine_lr(step, a.steps, a.warmup, a.lr)
        for g in opt.param_groups:
            g["lr"] = lr

        opt.zero_grad(set_to_none=True)
        total = 0.0
        for _ in range(a.accum):
            try:
                x, y = next(it)
            except StopIteration:
                it = iter(dl)
                x, y = next(it)
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with autocast:
                _, loss = model(x, y)
            (loss / a.accum).backward()
            total += loss.item() / a.accum

        if a.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
        opt.step()
        losses.append(total)

        if (step + 1) % a.log_every == 0:
            el = time.time() - t0
            done = step + 1 - start_step
            tps = done * tokens_per_step / el
            eta = (a.steps - step - 1) * tokens_per_step / max(1.0, tps)
            avg = sum(losses[-a.log_every:]) / len(losses[-a.log_every:])
            ppl = math.exp(min(20, avg))
            mem = torch.cuda.max_memory_allocated() / 1024**3 if dev == "cuda" else 0
            print(f"step {step+1:>6}/{a.steps} | loss {avg:.4f} | ppl {ppl:8.1f} | "
                  f"lr {lr:.2e} | {tps/1000:.1f}k tok/s | {mem:.1f}GB | ETA {fmt_time(eta)}",
                  flush=True)

        if (step + 1) % a.save_every == 0 or step + 1 == a.steps:
            save_ckpt(ck_path, model, opt, step + 1, cfg, tokens_seen=(step + 1) * tokens_per_step)
            print(f"  已保存 {ck_path} (step {step+1})", flush=True)

    dump_json(ROOT / "out/pretrain_summary.json", {
        "steps": a.steps, "tokens": a.steps * tokens_per_step,
        "final_loss": sum(losses[-50:]) / min(50, len(losses)),
        "params": model.num_params(), "elapsed_sec": time.time() - t0,
    })
    print("预训练完成。")


if __name__ == "__main__":
    main()
