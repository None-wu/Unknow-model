"""一条命令跑完全部训练，断了自动续。

    python train_all.py

阶段：
  1. 分词化 800M token 预训练流
  2. 预训练 24000 步（约 786M token，失败自动 --resume 重试）
  3. SFT 3 轮（工具轨迹 + 通用对话混训）
  4. 留出集评测

所有输出同时写 out/train_all.log，便于长时间后台运行时查看进度。
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOG = ROOT / "out" / "train_all.log"
PY = sys.executable

TARGET_TOKENS = 800_000_000
PRETRAIN_STEPS = 24000
SFT_EPOCHS = 3


def log(msg: str) -> None:
    line = f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(args: list[str], tag: str) -> None:
    log(f"▶ {tag}: {' '.join(args)}")
    t0 = time.time()
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    with LOG.open("a", encoding="utf-8") as f:
        p = subprocess.Popen(args, cwd=ROOT, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, env=env, bufsize=0)
        assert p.stdout is not None
        for raw in iter(p.stdout.readline, b""):
            text = raw.decode("utf-8", "replace").rstrip()
            if text:
                print(text, flush=True)
                f.write(text + "\n")
                f.flush()
        code = p.wait()
    dt = time.time() - t0
    if code != 0:
        raise RuntimeError(f"{tag} 失败，退出码 {code}，耗时 {dt/60:.1f} 分钟")
    log(f"✔ {tag} 完成，耗时 {dt/60:.1f} 分钟")


def stage_tokenize() -> None:
    bin_path = ROOT / "data/tokenized/pretrain.bin"
    have = bin_path.stat().st_size // 2 if bin_path.exists() else 0
    target = TARGET_TOKENS
    for attempt in range(1, 4):
        if have >= target * 0.95:
            log(f"分词化就绪：{have/1e6:.0f}M token")
            return
        log(f"当前 token 流 {have/1e6:.0f}M，目标 {target/1e6:.0f}M（第 {attempt} 次）")
        run([PY, "-u", "-m", "trainer.data", "--mode", "pretrain",
             "--target-tokens", str(target)], f"阶段1 分词化（第 {attempt} 次）")
        have = bin_path.stat().st_size // 2 if bin_path.exists() else 0
        if have >= target * 0.95:
            log(f"分词化就绪：{have/1e6:.0f}M token")
            return
        # 抽样步长估偏了：按实际产出比例把目标放大，下一次 stride 会自动变小
        target = int(target * target / max(1, have) * 1.02)
        log(f"只产出 {have/1e6:.0f}M，把目标上调到 {target/1e6:.0f}M 重跑")
    raise RuntimeError("分词化三次仍未达标")


def stage_pretrain() -> None:
    base = ROOT / "checkpoints/base.pt"
    if base.exists():
        from trainer.utils import load_ckpt  # 延迟导入，避免无谓的 torch 初始化
        _, ck, _ = load_ckpt(base)
        if ck.get("step", 0) >= PRETRAIN_STEPS:
            log(f"跳过预训练：base.pt 已到 step {ck['step']}")
            return

    args = [PY, "-u", "-m", "trainer.pretrain",
            "--steps", str(PRETRAIN_STEPS), "--batch", "16", "--accum", "4",
            "--seq-len", "512", "--lr", "6e-4", "--warmup", "600",
            "--save-every", "500", "--log-every", "20",
            "--out", "checkpoints/base.pt"]
    for attempt in range(1, 6):
        resume = base.exists()
        cmd = args + (["--resume"] if resume else [])
        if resume:
            log(f"检测到 {base.name}，第 {attempt} 次尝试将断点续跑")
        try:
            run(cmd, f"阶段2 预训练（第 {attempt} 次）")
            return
        except RuntimeError as e:
            log(f"⚠ {e}")
            if attempt == 5:
                raise
            time.sleep(10)


def stage_sft() -> None:
    out = ROOT / "checkpoints/sft.pt"
    run([PY, "-u", "-m", "trainer.sft",
         "--base", "checkpoints/base.pt", "--epochs", str(SFT_EPOCHS),
         "--batch", "8", "--accum", "2", "--lr", "1.2e-4",
         "--log-every", "100", "--eval-every", "600",
         "--out", str(out.relative_to(ROOT)).replace("\\", "/")], "阶段3 SFT")


def stage_eval() -> None:
    run([PY, "-u", "-m", "agent.eval", "--ckpt", "checkpoints/sft.pt",
         "--data", "data/tool_eval.jsonl", "--n", "200",
         "--dump", "out/eval_errors.jsonl"], "阶段4 评测")


def main() -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    log("=" * 70)
    log(f"全流程开始 | 目标 {TARGET_TOKENS/1e6:.0f}M token 预训练 + {SFT_EPOCHS} 轮 SFT")
    t0 = time.time()
    stage_tokenize()
    stage_pretrain()
    stage_sft()
    stage_eval()
    log(f"全部完成，总耗时 {(time.time()-t0)/3600:.2f} 小时")


if __name__ == "__main__":
    main()
