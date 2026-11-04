"""准备语料：下载 / 校验 / 提取纯文本缓存。

语料不随仓库分发（约 25GB，且上游为 CC-BY-NC 授权），本脚本从 HuggingFace
拉取 MiniMind 数据集并提取成训练所需的 `_corpus_cache/*.txt`。

三档规模，按磁盘和耐心选：

    python scripts/prepare_corpus.py --tier sample   # 仓库自带样本，秒级，用于冒烟
    python scripts/prepare_corpus.py --tier mini     # 3.0 GB 下载，1 小时级全流程
    python scripts/prepare_corpus.py --tier full     # 24 GB 下载，复现正式结果

产物：
    data/pretrain_t2t*.jsonl          原始预训练语料（--tier sample 时复制样本）
    data/sft_t2t_mini.jsonl           对话语料（SFT 混训用）
    data/lora_medical.jsonl           医学对话（SFT 混训用，full/mini 才有）
    _corpus_cache/*_extracted.txt     纯文本缓存（一行一条，词表训练与分词都读它）
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "_corpus_cache"
SAMPLES = ROOT / "samples" / "corpus"

REPO_ID = "jingyaogong/minimind_dataset"
REPO_URL = f"https://huggingface.co/datasets/{REPO_ID}"

# 文件名 -> 体积（仅用于提示，不参与校验）
FILES = {
    "pretrain_t2t.jsonl": "8.3 GB",
    "pretrain_t2t_mini.jsonl": "1.2 GB",
    "sft_t2t.jsonl": "14.1 GB",
    "sft_t2t_mini.jsonl": "1.7 GB",
    "lora_medical.jsonl": "34 MB",
}

TIERS = {
    "sample": [],
    "mini": ["pretrain_t2t_mini.jsonl", "sft_t2t_mini.jsonl", "lora_medical.jsonl"],
    "full": list(FILES),
}


def extract_text(rec: dict) -> str | None:
    """把一条样本压成一行纯文本，与 `_corpus_cache` 现有格式保持一致。

    预训练语料取 text 字段；对话语料把每轮 content 用空格连起来
    （reasoning_content 不入库，避免把思维链喂进词表统计）。
    """
    if isinstance(rec.get("text"), str):
        return rec["text"].replace("\n", " ").replace("\r", " ").strip() or None
    msgs = rec.get("conversations") or rec.get("messages")
    if isinstance(msgs, list):
        parts = []
        for m in msgs:
            c = m.get("content") if isinstance(m, dict) else None
            if isinstance(c, str) and c.strip():
                parts.append(c.replace("\n", " ").replace("\r", " ").strip())
        return " ".join(parts) or None
    return None


def extract_jsonl(src: Path, dst: Path, every: int = 200_000) -> int:
    """流式提取，避免把 GB 级文件读进内存。"""
    if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime and dst.stat().st_size > 0:
        print(f"  已存在，跳过: {dst.name} ({dst.stat().st_size / 1e9:.2f} GB)")
        return 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    n = bad = 0
    t0 = time.time()
    with src.open(encoding="utf-8", errors="replace") as fi, \
            dst.open("w", encoding="utf-8") as fo:
        for line in fi:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            text = extract_text(rec)
            if not text or len(text) < 12:
                continue
            fo.write(text + "\n")
            n += 1
            if every and n % every == 0:
                print(f"    {n / 1e6:.2f}M 行 | {time.time() - t0:.0f}s", flush=True)
    size = dst.stat().st_size / 1e9
    print(f"  {src.name} -> {dst.name}: {n} 行 / {size:.2f} GB"
          + (f"（{bad} 行解析失败）" if bad else ""))
    return n


def download(name: str) -> Path:
    dst = ROOT / "data" / name
    if dst.exists() and dst.stat().st_size > 0:
        print(f"  已存在，跳过下载: {name} ({dst.stat().st_size / 1e9:.2f} GB)")
        return dst
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise SystemExit("缺少 huggingface_hub，先装依赖：pip install -r requirements.txt") from None
    print(f"  下载 {name}（{FILES.get(name, '?')}）…若网络受限可换镜像："
          f"export HF_ENDPOINT=https://hf-mirror.com")
    got = hf_hub_download(repo_id=REPO_ID, filename=name, repo_type="dataset",
                          local_dir=str(ROOT / "data"))
    got = Path(got)
    if got != dst:  # 新版 hub 会落到子目录
        shutil.move(str(got), dst)
    print(f"  完成 {name} ({dst.stat().st_size / 1e9:.2f} GB)")
    return dst


def tier_sample() -> None:
    """把仓库自带的样本铺成标准文件名，让整条流水线能在几分钟内空跑一遍。"""
    pairs = [
        ("pretrain_sample.jsonl", "pretrain_t2t_mini.jsonl"),
        ("sft_sample.jsonl", "sft_t2t_mini.jsonl"),
    ]
    for s, d in pairs:
        src, dst = SAMPLES / s, ROOT / "data" / d
        if not src.exists():
            raise SystemExit(f"样本缺失：{src}（仓库应自带 samples/corpus/）")
        if not dst.exists() or dst.stat().st_size < src.stat().st_size:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
            print(f"  铺设样本 {src.name} -> data/{d} ({src.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    ap = argparse.ArgumentParser(description="下载语料并生成 _corpus_cache 纯文本缓存")
    ap.add_argument("--tier", choices=list(TIERS), default="mini",
                    help="sample=用仓库样本冒烟；mini=3GB；full=24GB（复现正式结果）")
    ap.add_argument("--skip-download", action="store_true", help="只做提取，假定 jsonl 已就位")
    a = ap.parse_args()

    CACHE.mkdir(parents=True, exist_ok=True)
    print(f"语料来源: {REPO_URL}")
    print(f"规模档位: {a.tier}\n")

    if a.tier == "sample":
        tier_sample()
    elif not a.skip_download:
        for name in TIERS[a.tier]:
            download(name)

    print("\n提取纯文本缓存（分词与词表训练都读这里）：")
    got = 0
    for name in TIERS.get(a.tier, TIERS["full"]) or []:
        src = ROOT / "data" / name
        if not src.exists():
            print(f"  跳过（不存在）: {name}")
            continue
        dst = CACHE / (src.stem + "_extracted.txt")
        got += extract_jsonl(src, dst) or 0

    if a.tier == "sample":
        # 样本档也要保证产物文件名齐全，否则 data.py 找不到文件
        for name in ["pretrain_t2t_mini.jsonl", "sft_t2t_mini.jsonl"]:
            src = ROOT / "data" / name
            if src.exists():
                dst = CACHE / (src.stem + "_extracted.txt")
                got += extract_jsonl(src, dst) or 0

    total = sum(p.stat().st_size for p in CACHE.glob("*.txt"))
    print(f"\n_corpus_cache 共 {total / 1e9:.2f} GB")
    if got == 0 and total == 0:
        sys.exit("没有产出任何语料，检查下载是否失败")


if __name__ == "__main__":
    main()
