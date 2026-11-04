"""从大语料里等距抽样，产出一份用于训练词表的混合文本。

为什么要抽样而不是全量：
- 19GB 纯文本按 ~1.5 chars/token 算有 120 亿 token，40M 模型只需要 8~10 亿；
- 词表训练只需要几百 MB 就能收敛到与全量几乎相同的 merges；
- 关键是"配比"而不是"总量"：通用中文 + 对话 + 工具调用语法 都要有。

抽样方式：按行等距（stride），避免只吃到文件开头。
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "_corpus_cache"


def stride_sample(path: Path, target_mb: float, rng: random.Random) -> list[str]:
    """按字节量等距抽行。先估文件大小，算出步长，再单遍扫描。"""
    target = int(target_mb * 1024 * 1024)
    size = path.stat().st_size
    if size == 0:
        return []
    # 平均行长约 size/行数，无法预知，这里用 400 字节估一个保守步长
    est_lines = max(1, size // 400)
    stride = max(1, est_lines // max(1, target // 400))
    out: list[str] = []
    taken = 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if i % stride:
                continue
            line = line.strip()
            if len(line) < 12:
                continue
            out.append(line)
            taken += len(line.encode("utf-8"))
            if taken >= target:
                break
    rng.shuffle(out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/_tok_sample.txt")
    ap.add_argument("--total-mb", type=float, default=320.0, help="目标总字节数")
    ap.add_argument("--seed", type=int, default=7)
    # 配比：通用预训练语料 45%，对话语料 40%，工具轨迹 15%
    ap.add_argument("--mix", default="45,40,15")
    a = ap.parse_args()
    rng = random.Random(a.seed)

    p_gen, p_chat, p_tool = (float(x) for x in a.mix.split(","))
    parts = {
        "gen": [(CACHE / "pretrain_t2t_extracted.txt", p_gen * 0.6),
                (CACHE / "pretrain_t2t_mini_extracted.txt", p_gen * 0.4)],
        "chat": [(CACHE / "sft_t2t_extracted.txt", p_chat * 0.6),
                 (CACHE / "sft_t2t_mini_extracted.txt", p_chat * 0.4)],
    }

    out_path = ROOT / a.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for name, items in parts.items():
            for p, mb in items:
                if not p.exists():
                    print(f"跳过（不存在）: {p}")
                    continue
                lines = stride_sample(p, mb / 100 * a.total_mb, rng)
                blob = "\n".join(lines)
                f.write(blob + "\n")
                total += len(blob.encode("utf-8"))
                print(f"  {name:5s} {p.name:34s} {len(blob)/1e6:7.1f} MB  {len(lines):>7} 行")

        # 工具轨迹：源文件小，重复多轮以让命令语法在词表里有足够权重
        tool_file = ROOT / "data" / "tool_sft.jsonl"
        if tool_file.exists():
            target = int(p_tool / 100 * a.total_mb * 1e6)
            raw = tool_file.read_text(encoding="utf-8")
            reps = max(1, target // max(1, len(raw.encode("utf-8"))))
            reps = min(reps, 40)
            f.write(raw * reps)
            total += len(raw.encode("utf-8")) * reps
            print(f"  tool  {tool_file.name:34s} {len(raw)*reps/1e6:7.1f} MB  x{reps} 轮")

    print(f"\n总字节 {total/1e6:.1f} MB -> {out_path}")


if __name__ == "__main__":
    main()
