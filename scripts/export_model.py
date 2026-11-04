"""导出推理用检查点：去掉优化器状态，体积缩到 1/3。

训练检查点里带 AdamW 的 exp_avg / exp_avg_sq（各占一份参数），
`checkpoints/sft.pt` 因此 712MB；推理只需要权重（约 250MB）。

    python scripts/export_model.py --ckpt checkpoints/sft.pt --out release/unknow-62m.pt

导出后仍是本项目自己的格式（config + model 两个键），
`run.py chat --ckpt release/unknow-62m.pt` 可直接加载。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/sft.pt")
    ap.add_argument("--out", default="release/unknow-62m.pt")
    a = ap.parse_args()

    src = ROOT / a.ckpt
    ck = torch.load(src, map_location="cpu", weights_only=False)
    slim = {
        "model": ck["model"],
        "step": ck.get("step"),
        "config": ck["config"],
        "val_loss": ck.get("val_loss"),
        "note": "inference-only export (optimizer stripped)",
    }
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(slim, out)
    print(f"{src.name} {src.stat().st_size / 1e6:.0f} MB -> "
          f"{out.name} {out.stat().st_size / 1e6:.0f} MB")
    # 自检：导出的东西能被 load_ckpt 正常读回
    from trainer.utils import load_ckpt
    model, _, cfg = load_ckpt(out)
    print(f"校验通过：{model.num_params() / 1e6:.2f}M 参数 | dim {cfg.dim} | {cfg.n_layers} 层")


if __name__ == "__main__":
    main()
