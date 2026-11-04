"""数据管线：预训练 token 流 + SFT 样本（带 loss mask）。

两个关键决定：
1. 预训练用 uint16 memmap 单流，随机窗口采样——40M 模型不需要文档边界，
   但每篇之间插 </s>，让模型学会"该结束了"；
2. SFT 只对 assistant 段计 loss。user 和 tool_result 是上下文，
   如果也计 loss，模型会学着去预测命令输出，那是在浪费容量。
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from tools.schema import (A_TOKEN, CALL_TOKEN, END_TOKEN, RESULT_TOKEN,
                          T_TOKEN, U_TOKEN)

ROOT = Path(__file__).resolve().parent.parent
IGNORE = -100


# ---------------------------------------------------------------------------
# 预训练
# ---------------------------------------------------------------------------
def count_lines(path: Path, limit: int | None = None) -> int:
    n = 0
    with open(path, "rb") as f:
        for _ in f:
            n += 1
            if limit and n >= limit:
                break
    return n


def write_pretrain_stream(paths: list[Path], tok, out: Path, target_tokens: int,
                          log_every: int = 200) -> int:
    """把多份纯文本按比例抽样、编码成 uint16 流。"""
    sizes = [p.stat().st_size for p in paths]
    total_size = sum(sizes)
    out.parent.mkdir(parents=True, exist_ok=True)
    eos = tok.convert_tokens_to_ids("</s>")

    written = 0
    fout = open(out, "wb")
    try:
        for p, size in zip(paths, sizes):
            quota = int(target_tokens * size / total_size)
            # 等距抽样：不从头顺读，避免只吃到文件开头那几百万行。
            # 中文 UTF-8 约 4.7 字节/token（3 字节/字 ÷ 1.9 字/token），
            # 这里按 6 字节估，宁可少跳几行也別抽不满配额；
            # 抽不满时会退化成整份读完，由调用方按实际产出再补一次。
            target_bytes = max(1, int(quota * 6.0))
            stride = max(1, size // target_bytes)
            got = 0
            last_log = 0
            buf: list[str] = []
            with open(p, "r", encoding="utf-8", errors="replace") as f:
                for li, line in enumerate(f):
                    if stride > 1 and li % stride:
                        continue
                    line = line.strip()
                    if len(line) < 8:
                        continue
                    buf.append(line)
                    if len(buf) >= 1000:
                        ids = tok(buf)["input_ids"]
                        arr = np.concatenate(
                            [np.array(x, dtype=np.uint16).tolist() + [eos] for x in ids]
                        ).astype(np.uint16)
                        fout.write(arr.tobytes())
                        got += len(arr)
                        written += len(arr)
                        buf = []
                        if got >= quota:
                            break
                        if written - last_log >= 20_000_000:
                            last_log = written
                            print(f"  已写入 {written/1e6:.0f}M tokens ({p.name})", flush=True)
            if buf and got < quota:
                ids = tok(buf)["input_ids"]
                arr = np.concatenate(
                    [np.array(x, dtype=np.uint16).tolist() + [eos] for x in ids]
                ).astype(np.uint16)
                fout.write(arr.tobytes())
                written += len(arr)
            print(f"  {p.name}: {got/1e6:.2f}M tokens", flush=True)
    finally:
        fout.close()
    print(f"总 token: {written/1e6:.2f}M -> {out} ({out.stat().st_size/1e9:.2f} GB)")
    return written


class PretrainDataset(Dataset):
    def __init__(self, bin_path: str | Path, seq_len: int, num_samples: int):
        self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.seq_len = seq_len
        self.num_samples = num_samples

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, i: int):
        hi = len(self.data) - self.seq_len - 1
        s = random.randint(0, hi)
        chunk = self.data[s:s + self.seq_len + 1].astype(np.int64)
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        return x, y


# ---------------------------------------------------------------------------
# SFT 组装
# ---------------------------------------------------------------------------
def _seg(tok, text: str, trainable: bool) -> tuple[list[int], list[int]]:
    ids = tok(text, add_special_tokens=False)["input_ids"]
    labels = list(ids) if trainable else [IGNORE] * len(ids)
    return ids, labels


def build_trajectory(tok, rec: dict) -> tuple[list[int], list[int]] | None:
    """{"events":[...]} -> (input_ids, labels)

    协议：user / assistant / tool 三段式。assistant 的两类内容（tool_call 与
    最终回答）都可训练；tool_result 只作为条件。
    """
    ids: list[int] = []
    labels: list[int] = []

    def add(text: str, trainable: bool) -> None:
        i, l = _seg(tok, text, trainable)
        ids.extend(i)
        labels.extend(l)

    for ev in rec["events"]:
        t = ev["t"]
        if t == "user":
            add(f"{U_TOKEN}{ev['text']}{END_TOKEN}", False)
        elif t == "call":
            payload = json.dumps(
                {"name": ev["name"], "args": ev["args"]},
                ensure_ascii=False, separators=(",", ":"),
            )
            add(f"{A_TOKEN}{CALL_TOKEN}{payload}{END_TOKEN}", True)
            add(f"{T_TOKEN}{RESULT_TOKEN}{ev['output']}{END_TOKEN}", False)
        elif t == "assistant":
            add(f"{A_TOKEN}{ev['text']}{END_TOKEN}", True)
    if not ids:
        return None
    return ids, labels


def build_chat(tok, msgs: list[dict], max_turns: int = 8) -> tuple[list[int], list[int]] | None:
    """把常规对话数据转成同一套协议，保持"会聊天"的能力。"""
    ids: list[int] = []
    labels: list[int] = []
    for m in msgs[:max_turns]:
        role = m.get("role")
        content = (m.get("content") or "").strip()
        if not content or role not in ("user", "assistant"):
            continue
        text = f"{U_TOKEN if role == 'user' else A_TOKEN}{content}{END_TOKEN}"
        i, l = _seg(tok, text, role == "assistant")
        ids.extend(i)
        labels.extend(l)
    if not ids or all(l == IGNORE for l in labels):
        return None
    return ids, labels


class SFTDataset(Dataset):
    """注意：labels 必须相对输入右移一位（logits[t] 预测 ids[t+1]）。
    不做这一步，模型学到的是"复制当前 token"，loss 会迅速归零但完全不会生成。
    """

    def __init__(self, items: list[tuple[list[int], list[int]]], max_len: int = 768):
        prepared = []
        for ids, labels in items:
            ids, labels = ids[:max_len], labels[:max_len]
            if len(ids) < 3:
                continue
            # 右移：用位置 t 的输出去预测位置 t+1 的标签
            prepared.append((ids[:-1], labels[1:]))
        self.items = prepared

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        x, y = self.items[i]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


def collate(batch, pad_id: int = 0):
    max_len = max(x.size(0) for x, _ in batch)
    bx = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    by = torch.full((len(batch), max_len), IGNORE, dtype=torch.long)
    for i, (x, y) in enumerate(batch):
        bx[i, : x.size(0)] = x
        by[i, : y.size(0)] = y
    return bx, by


# ---------------------------------------------------------------------------
# CLI：一次性把数据准备好
# ---------------------------------------------------------------------------
def main() -> None:
    import argparse

    from transformers import PreTrainedTokenizerFast

    ap = argparse.ArgumentParser(description="准备预训练 token 流与 SFT 样本")
    ap.add_argument("--tokenizer", default="tokenizer/tokenizer_v3")
    ap.add_argument("--mode", choices=["pretrain", "sft", "both"], default="both")
    ap.add_argument("--target-tokens", type=int, default=800_000_000)
    ap.add_argument("--chat-ratio", type=float, default=0.28)
    ap.add_argument("--tool-data", default="data/tool_sft_final.jsonl",
                    help="工具轨迹 jsonl；未跑 extra_sft.py 时可用 data/tool_sft.jsonl")
    ap.add_argument("--max-len", type=int, default=768)
    ap.add_argument("--seed", type=int, default=13)
    a = ap.parse_args()

    tok = PreTrainedTokenizerFast.from_pretrained(str(ROOT / a.tokenizer))
    print(f"词表: {tok.vocab_size}")

    if a.mode in ("pretrain", "both"):
        paths = [
            ROOT / "_corpus_cache" / "pretrain_t2t_extracted.txt",
            ROOT / "_corpus_cache" / "pretrain_t2t_mini_extracted.txt",
            ROOT / "_corpus_cache" / "sft_t2t_extracted.txt",
            ROOT / "_corpus_cache" / "sft_t2t_mini_extracted.txt",
        ]
        paths = [p for p in paths if p.exists()]
        if not paths:
            raise SystemExit("找不到预训练语料缓存，请先跑 tokenizer/sample_corpus.py")
        write_pretrain_stream(paths, tok, ROOT / "data/tokenized/pretrain.bin", a.target_tokens)

    if a.mode in ("sft", "both"):
        rng = random.Random(a.seed)
        items: list[tuple[list[int], list[int]]] = []

        tool_path = ROOT / a.tool_data
        n_tool = 0
        if tool_path.exists():
            for line in tool_path.open(encoding="utf-8"):
                rec = json.loads(line)
                r = build_trajectory(tok, rec)
                if r:
                    items.append(r)
                    n_tool += 1
        print(f"工具轨迹: {n_tool} 条")

        chat_target = int(n_tool * a.chat_ratio / max(1e-6, 1 - a.chat_ratio))
        chat_files = [ROOT / "data" / "sft_t2t_mini.jsonl", ROOT / "data" / "lora_medical.jsonl"]
        n_chat = 0
        for cf in chat_files:
            if not cf.exists() or n_chat >= chat_target:
                continue
            want = chat_target // len(chat_files)
            with cf.open(encoding="utf-8") as f:
                for line in f:
                    if n_chat >= want and cf.name.endswith("mini.jsonl"):
                        break
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    msgs = rec.get("conversations") or rec.get("messages")
                    if not msgs:
                        continue
                    r = build_chat(tok, msgs)
                    if r:
                        items.append(r)
                        n_chat += 1
                    if n_chat >= chat_target:
                        break
        print(f"对话样本: {n_chat} 条")

        rng.shuffle(items)
        lens = [len(x) for x, _ in items]
        print(f"合计 {len(items)} 条 | 平均长度 {sum(lens)/len(lens):.0f} | "
              f"P95 {sorted(lens)[int(len(lens)*0.95)]} | 最长 {max(lens)}")
        ds = SFTDataset(items, a.max_len)
        out = ROOT / "data/tokenized/sft.pt"
        out.parent.mkdir(parents=True, exist_ok=True)
        # 注意：必须存"未右移"的原始 (ids, labels)。
        # SFTDataset 在 __getitem__ 时才做右移；如果这里存 ds.items（已右移），
        # 训练时 SFTDataset 会再移一次，模型变成"预测下下个 token"——
        # loss 看着在降，生成却会跳字（{"":topf",args{"":path"}} 这种）。
        raw = [(ids[:a.max_len], lab[:a.max_len]) for ids, lab in items]
        torch.save({"items": raw, "pad_id": tok.pad_token_id or 0}, out)
        print(f"已保存 {out} ({out.stat().st_size/1e6:.1f} MB)")

        # 打印一条还原样本，肉眼确认协议与掩码
        x, y = ds[0]
        print("\n--- 样本还原 ---")
        print(tok.decode(x.tolist())[:600])
        print("--- 训练目标（仅 assistant 段）---")
        keep = [t for t, l in zip(x.tolist(), y.tolist()) if l != IGNORE]
        print(tok.decode(keep)[:300])


if __name__ == "__main__":
    main()
