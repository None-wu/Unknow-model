"""生成留出评测集：与训练集**不同随机种子**、且指令文本不重复。

为什么单独一个脚本：训练集是 `run.py synth` 造的，如果评测也用它，
同一个模板换个文件就成了"同源评测"，指标会虚高。这里用独立种子的
分片重新合成，并剔除与训练集完全相同的用户指令，保证是真正的留出集。

    python scripts/make_eval.py --n 500 --out data/tool_eval.jsonl

注意：这是"同分布留出"（模板相同、参数不同），只衡量指令→工具的映射是否
稳定，不代表真实口语泛化。口语泛化看 `run.py probe`。
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.executor import BashWorker, Sandbox  # noqa: E402
from tools.synth import one  # noqa: E402

SEED_BASE = 90210


def user_text(rec: dict) -> str:
    for e in rec.get("events", []):
        if e.get("t") == "user":
            return e.get("text", "")
    return ""


def shard(args: tuple[int, int]) -> list[str]:
    """一个分片生成 n 条候选，返回 jsonl 文本行。"""
    seed, n = args
    rng = random.Random(seed)
    worker = BashWorker().start()
    clean = Sandbox(worker, reuse=True)
    clean.__enter__()
    out: list[str] = []
    try:
        while len(out) < n:
            tr = one(rng, clean, lambda: Sandbox(worker))
            if tr is None:
                continue
            if tr.steps and not any(s.ok for s in tr.steps):
                continue
            out.append(json.dumps(tr.to_record(), ensure_ascii=False))
    finally:
        clean.destroy()
        worker.close()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="生成留出评测集（独立种子 + 去重）")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--out", default="data/tool_eval.jsonl")
    ap.add_argument("--train", default="data/tool_sft_final.jsonl",
                    help="训练集路径，用于剔除重复指令")
    ap.add_argument("--workers", type=int, default=0)
    a = ap.parse_args()

    seen: set[str] = set()
    train = ROOT / a.train
    if train.exists():
        with train.open(encoding="utf-8") as f:
            for line in f:
                try:
                    seen.add(user_text(json.loads(line)))
                except json.JSONDecodeError:
                    continue
        print(f"训练集 {train.name}：{len(seen)} 条指令已登记为不可用于评测")
    else:
        print(f"未找到训练集 {train}，跳过去重（评测集可能与训练重叠）")

    workers = a.workers or max(1, (mp.cpu_count() or 2) - 1)
    # 多要 40% 余量，被去重和重复指令消耗掉一部分
    per = int(a.n / workers * 1.4) + 1
    jobs = [(SEED_BASE + i * 7919, per) for i in range(workers + 2)]

    kept: list[str] = []
    dup = 0
    with mp.get_context("spawn").Pool(workers) as pool:
        for i, chunk in enumerate(pool.imap_unordered(shard, jobs), 1):
            for line in chunk:
                if len(kept) >= a.n:
                    continue
                t = user_text(json.loads(line))
                if not t or t in seen:
                    dup += 1
                    continue
                seen.add(t)
                kept.append(line)
            print(f"  分片 {i}/{len(jobs)}：累计 {len(kept)} 条（丢弃重复 {dup}）", flush=True)
            if len(kept) >= a.n:
                break

    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(kept) + "\n", encoding="utf-8")
    kinds: dict[str, int] = {}
    for line in kept:
        k = json.loads(line).get("kind", "?")
        kinds[k] = kinds.get(k, 0) + 1
    print(f"\n已写 {out}：{len(kept)} 条 | 分布 {kinds} | 丢弃重复 {dup}")


if __name__ == "__main__":
    main()
