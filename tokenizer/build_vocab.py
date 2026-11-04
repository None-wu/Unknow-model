"""词表构建入口：固定协议 token，避免手敲参数出错。

踩过的坑：在 Git Bash 里直接传 `--special-tokens '</s>'`，
MSYS 会把 "</s>" 当路径转成 <C:/.../s> 塞进词表，
最终 eos 缺失、id 错位。所以协议 token 写在这里，不经过 shell。
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

# 与 tools/schema.py 的 PROTOCOL_TOKENS 必须一致
SPECIAL_TOKENS = [
    "<s>", "</s>", "<unk>", "<pad>", "<mask>",
    "<|user|>", "<|assistant|>", "<|tool|>",
    "<|tool_call|>", "<|tool_result|>", "<|end|>", "<|think|>",
]


def load_trainer():
    spec = importlib.util.spec_from_file_location("train_tokenizer", HERE / "train_tokenizer.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/_tok_sample.txt")
    ap.add_argument("--vocab-size", type=int, default=16384)
    ap.add_argument("--out-dir", default="tokenizer/tokenizer_v3")
    ap.add_argument("--min-freq", type=int, default=2)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--eval-samples", type=int, default=300)
    a = ap.parse_args()

    corpus = ROOT / a.corpus
    if not corpus.exists():
        raise SystemExit(f"找不到语料 {corpus}，先跑 python -m tokenizer.sample_corpus")

    sys.argv = [
        "train_tokenizer.py",
        "--corpus", str(corpus),
        "--vocab-size", str(a.vocab_size),
        "--out-dir", str(ROOT / a.out_dir),
        "--min-freq", str(a.min_freq),
        "--threads", str(a.threads),
        "--eval-samples", str(a.eval_samples),
        "--special-tokens", *SPECIAL_TOKENS,
    ]
    load_trainer().main()
    verify(ROOT / a.out_dir)


def verify(out_dir: Path) -> None:
    import json

    from transformers import PreTrainedTokenizerFast

    tok = PreTrainedTokenizerFast.from_pretrained(str(out_dir))
    j = json.loads((out_dir / "tokenizer.json").read_text(encoding="utf-8"))
    vocab = j["model"]["vocab"]
    problems = []
    for t in SPECIAL_TOKENS:
        i = vocab.get(t)
        if i is None:
            problems.append(f"{t} 不在词表")
        elif i >= len(vocab):
            problems.append(f"{t} id 越界")
    if tok.eos_token_id is None:
        problems.append("eos 未设置")
    print(f"\n词表 {len(vocab)}，协议 token 检查：{'通过' if not problems else problems}")
    for t in SPECIAL_TOKENS:
        print(f"  {t:16s} id={tok.convert_tokens_to_ids(t)}")
    print(f"  eos={tok.eos_token} bos={tok.bos_token} pad={tok.pad_token}")


if __name__ == "__main__":
    main()
