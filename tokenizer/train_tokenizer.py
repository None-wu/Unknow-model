#!/usr/bin/env python
"""生产级 Byte-Level BPE 分词器训练脚本（HuggingFace tokenizers）。

特性：
- 流式读取语料（逐行 yield），支持多文件 glob，不一次性载入内存，可吃百 MB~GB 级语料
- GPT-2 / RoBERTa 风格 ByteLevel 预分词（byte-level 天然无 OOV）
- 特殊 token 预留（BOS/EOS/UNK/PAD/MASK），训练与加载配置一致
- 训练后自动评估：词表规模、平均 token 数、压缩率、encode/decode 往返一致性
- 输出：tokenizer.json（tokenizers 格式）+ 可直接被 Trainer 使用的 PreTrainedTokenizerFast 目录
- 生产要素：日志、耗时统计、异常处理、参数化配置

用法：
    python train_tokenizer.py --corpus "data/*.txt" --vocab-size 32000 \
        --out-dir ./tokenizer --min-freq 2 --threads 4

依赖：tokenizers（transformers 自带）、transformers
"""
from __future__ import annotations

# ---- 必须在导入 tokenizers / transformers 之前设置，否则可能不生效 ----
import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import glob
import json
import logging
import time
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("train_tokenizer")

# 与 GPT-2 一致的 byte-level 特殊字符：空格前缀、无 OOV
# add_prefix_space=False 为 GPT-2 原版行为：仅后续词加 "Ġ" 前缀，往返一致
BYTE_LEVEL_PRE = dict(add_prefix_space=False, use_regex=True)

# 常见特殊 token 候选名（用于把 --special-tokens 自动映射到 HF 字段）
SPECIAL_TOKEN_CANDIDATES: dict[str, list[str]] = {
    "bos_token": ["<s>", "<bos>", "[BOS]"],
    "eos_token": ["</s>", "<eos>", "[EOS]"],
    "unk_token": ["<unk>", "[UNK]"],
    "pad_token": ["<pad>", "[PAD]"],
    "mask_token": ["<mask>", "[MASK]"],
    "sep_token": ["<sep>", "[SEP]"],
    "cls_token": ["<cls>", "[CLS]"],
}


# ----------------------------------------------------------------------------
# Tokenizer 构建
# ----------------------------------------------------------------------------
def build_tokenizer() -> Tokenizer:
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(**BYTE_LEVEL_PRE)
    tok.decoder = decoders.ByteLevel()
    return tok


# ----------------------------------------------------------------------------
# 语料读取 / 展开
# ----------------------------------------------------------------------------
def is_json_corpus(path: str) -> bool:
    """判断语料文件是否为 JSONL / JSON 格式。"""
    return str(path).lower().endswith((".jsonl", ".json"))


def extract_line_text(line: str, text_field: str = "text") -> str | None:
    """从 JSONL/JSON 行中提取文本字段；非 JSON 或提取失败返回 None。

    兼容多种形态：
    - 对象形：{"text": "..."}，取 text_field 字段（字符串直接返回）
    - 对象形且 text_field 为列表（如 SFT 多轮对话）：
      逐条取 "content" 或字符串元素，用换行拼接
    - 对象形无 text_field 但有 "conversations"：按上述列表逻辑展平
    - 字符串形：JSON 裸字符串（json.loads 后为 str），直接作为文本
    """
    try:
        obj = json.loads(line)
    except Exception:
        return None

    if isinstance(obj, str):
        return obj
    if not isinstance(obj, dict):
        return None

    value = obj.get(text_field)
    if value is None and "conversations" in obj:
        value = obj["conversations"]

    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                content = item.get("content")
                if isinstance(content, str) and content.strip():
                    parts.append(content)
        return "\n".join(parts) if parts else None
    return None


def prepare_corpus(files: list[str], text_field: str, cache_dir: Path) -> list[str]:
    """将 JSONL/JSON 语料流式转换为纯文本缓存文件（供 tokenizers 训练）。

    - 纯文本文件（.txt 等）原样返回
    - JSON/JSONL 文件逐行解析并提取 text_field 字段，写入 cache_dir，
      已存在同名缓存则复用，避免重复转换
    - 返回列表为训练/评估实际使用的文件路径
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    prepared: list[str] = []
    for p in files:
        if not is_json_corpus(p):
            prepared.append(p)
            continue
        dst = cache_dir / (Path(p).stem + "_extracted.txt")
        if dst.exists():
            logger.info("复用 JSON 提取缓存: %s", dst)
            prepared.append(str(dst))
            continue
        logger.info("提取 JSON 文本字段: %s -> %s", p, dst)
        written = 0
        with open(p, "r", encoding="utf-8", errors="replace") as f_in, \
             open(dst, "w", encoding="utf-8") as f_out:
            for line in f_in:
                t = extract_line_text(line, text_field)
                if t:
                    f_out.write(t.replace("\n", " ").strip() + "\n")
                    written += 1
        logger.info("JSON 提取完成: %d 行 -> %s", written, dst)
        prepared.append(str(dst))
    return prepared


def iter_corpus(paths: list[str], text_field: str = "text"):
    """逐行流式 yield 非空行。仅用于评估抽样；训练走 tokenizers 内部流式读取。

    - JSONL/JSON 文件自动提取 text_field 字段，避免 JSON 结构噪音进样本
    - 使用 errors="replace" 是权衡：Python 文本模式必须先解码，
      遇到非法 UTF-8 字节只能替换；若要严格无损，请预处理语料为合法 UTF-8。
    """
    for p in paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"语料文件不存在: {p}")
        logger.info("读取语料: %s", p)
        is_json = is_json_corpus(p)
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.rstrip("\r\n")
                if is_json:
                    line = extract_line_text(line, text_field) or ""
                if line.strip():
                    yield line


def resolve_corpus_patterns(patterns: list[str]) -> list[str]:
    """展开 corpus 模式（支持 glob），返回去重后的文件绝对路径列表。"""
    files: list[str] = []
    for pat in patterns:
        matched = glob.glob(pat)
        if not matched:
            logger.warning("模式无匹配文件: %s", pat)
        for m in matched:
            if os.path.isfile(m):
                files.append(os.path.abspath(m))

    # 去重保序
    seen: set[str] = set()
    uniq: list[str] = []
    for f in files:
        if f not in seen:
            seen.add(f)
            uniq.append(f)

    if not uniq:
        raise FileNotFoundError("未找到任何语料文件，请检查 --corpus")
    return uniq


def count_lines(paths: list[str]) -> int:
    """粗略统计总行数（可选，--count-lines 打开）。二进制读更快。"""
    total = 0
    for p in paths:
        with open(p, "rb") as f:
            for _ in f:
                total += 1
    return total


def collect_eval_samples(files: list[str], limit: int, text_field: str = "text") -> list[str]:
    """抽取评估样本。

    - limit < 0：返回全部非空行
    - limit = 0：返回空列表
    - limit > 0：返回前 limit 行
    """
    if limit == 0:
        return []
    samples: list[str] = []
    for line in iter_corpus(files, text_field):
        samples.append(line)
        if 0 <= limit <= len(samples):
            break
    return samples


# ----------------------------------------------------------------------------
# 训练
# ----------------------------------------------------------------------------
def train(args: argparse.Namespace) -> Tokenizer:
    tok = build_tokenizer()
    files = resolve_corpus_patterns(args.corpus)

    # JSONL/JSON 语料先提取文本字段到纯文本缓存，避免 JSON 结构噪音进词表
    if any(is_json_corpus(f) for f in files):
        cache_dir = (
            Path(args.cache_dir)
            if args.cache_dir
            else Path(args.out_dir).parent / "_corpus_cache"
        )
        files = prepare_corpus(files, args.text_field, cache_dir)

    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_freq,
        special_tokens=args.special_tokens,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    t0 = time.time()
    tok.train(files, trainer=trainer)
    elapsed = time.time() - t0
    logger.info("训练完成，耗时 %.1fs，词表大小: %d", elapsed, tok.get_vocab_size())
    return tok


# ----------------------------------------------------------------------------
# 评估
# ----------------------------------------------------------------------------
def evaluate(tok: Tokenizer, sample_texts: list[str]) -> dict:
    """评估：token 数、压缩率、往返一致性。"""
    stats = {
        "samples": len(sample_texts),
        "roundtrip_ok": 0,
        "total_chars": 0,
        "total_tokens": 0,
    }
    for text in sample_texts:
        ids = tok.encode(text).ids
        decoded = tok.decode(ids)
        stats["total_chars"] += len(text)
        stats["total_tokens"] += len(ids)
        if decoded == text:
            stats["roundtrip_ok"] += 1
    stats["avg_chars_per_token"] = (
        stats["total_chars"] / stats["total_tokens"] if stats["total_tokens"] else 0.0
    )
    stats["roundtrip_rate"] = (
        stats["roundtrip_ok"] / stats["samples"] if stats["samples"] else 0.0
    )
    return stats


# ----------------------------------------------------------------------------
# 保存
# ----------------------------------------------------------------------------
def build_special_map(special_tokens: list[str]) -> dict[str, str]:
    """把 --special-tokens 自动映射到 HF 字段名。"""
    mapping: dict[str, str] = {}
    for field, candidates in SPECIAL_TOKEN_CANDIDATES.items():
        for name in candidates:
            if name in special_tokens:
                mapping[field] = name
                break
    return mapping


def save_pretrained(
    tok: Tokenizer,
    out_dir: Path,
    special_tokens: list[str],
    model_max_length: int | None = None,
) -> Path:
    """保存为 transformers 可直接使用的 PreTrainedTokenizerFast。"""
    out_dir.mkdir(parents=True, exist_ok=True)

    # tokenizers 训练后会重置 ByteLevel 配置，保存前显式恢复，
    # 确保 tokenizer.json 与训练配置一致
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(**BYTE_LEVEL_PRE)
    tok.decoder = decoders.ByteLevel()

    json_path = out_dir / "tokenizer.json"
    tok.save(str(json_path))

    special_map = build_special_map(special_tokens)
    missing = [
        f for f in ("bos_token", "eos_token", "unk_token", "pad_token", "mask_token")
        if f not in special_map
    ]
    if missing:
        logger.warning(
            "以下 HF 字段未能从 --special-tokens 推断，将保持默认: %s", missing
        )

    hf_kwargs: dict = dict(special_map)
    if model_max_length is not None:
        hf_kwargs["model_max_length"] = model_max_length

    hf_tok = PreTrainedTokenizerFast(
        tokenizer_file=str(json_path),
        **hf_kwargs,
    )
    hf_tok.save_pretrained(str(out_dir))

    logger.info(
        "已保存文件: %s",
        ", ".join(["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]),
    )
    return out_dir


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="生产级 Byte-Level BPE 分词器训练")
    p.add_argument(
        "--corpus", nargs="+", required=True,
        help="语料文件/glob 模式，支持多个，如 data/*.txt",
    )
    p.add_argument("--vocab-size", type=int, default=32000, help="目标词表大小（默认 32000）")
    p.add_argument("--min-freq", type=int, default=2, help="合并对最低出现频次（默认 2）")
    p.add_argument("--out-dir", type=str, default="./tokenizer", help="输出目录")
    p.add_argument(
        "--threads", type=int, default=0,
        help="Rayon 线程数（0 = 由 tokenizers 自动决定；>0 时在训练前设置 RAYON_NUM_THREADS）",
    )
    p.add_argument(
        "--eval-samples", type=int, default=200,
        help="评估样本数（<0 表示全部；0 表示不抽样；默认 200）",
    )
    p.add_argument(
        "--special-tokens", nargs="+",
        default=["<s>", "</s>", "<unk>", "<pad>", "<mask>"],
        help="特殊 token 列表（必须最先进入词表）",
    )
    p.add_argument(
        "--count-lines", action="store_true",
        help="统计语料总行数（会完整读一遍，大语料慎用）",
    )
    p.add_argument(
        "--text-field", type=str, default="text",
        help="JSONL/JSON 语料中的文本字段名（默认 text）",
    )
    p.add_argument(
        "--cache-dir", type=str, default=None,
        help="JSON 语料提取的纯文本缓存目录（默认 <out-dir> 的父目录/_corpus_cache）",
    )
    p.add_argument(
        "--model-max-length", type=int, default=None,
        help="写入 tokenizer_config.json 的 model_max_length（可选）",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # 允许通过 CLI 覆盖 Rayon 并行度；必须在任何 tokenizers 训练前设置
    if args.threads and args.threads > 0:
        os.environ["RAYON_NUM_THREADS"] = str(args.threads)

    files = resolve_corpus_patterns(args.corpus)
    logger.info("语料文件 %d 个", len(files))

    if args.count_lines:
        t_lines = count_lines(files)
        logger.info("总行数约 %d", t_lines)

    tok = train(args)

    actual = tok.get_vocab_size()
    if actual < args.vocab_size:
        logger.warning(
            "实际词表 %d 未达目标 %d（语料不足或 min_frequency 过高）",
            actual, args.vocab_size,
        )

    out_dir = Path(args.out_dir)
    save_pretrained(tok, out_dir, args.special_tokens, args.model_max_length)
    logger.info("tokenizer 已保存至: %s", out_dir.resolve())

    sample_texts = collect_eval_samples(files, args.eval_samples, args.text_field)
    if not sample_texts:
        sample_texts = [
            "hello world",
            "你好，世界",
            "The quick brown fox jumps over the lazy dog.",
        ]

    stats = evaluate(tok, sample_texts)
    logger.info(
        "评估: 样本 %d 个，往返一致率 %.1f%%，平均每 token 字符数 %.2f",
        stats["samples"],
        stats["roundtrip_rate"] * 100,
        stats["avg_chars_per_token"],
    )
    print(json.dumps({"vocab_size": actual, "eval": stats}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()