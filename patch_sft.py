"""补丁微调：修复「你是谁」类身份对话缺失。

问题：SFT 数据里只有拒答 + 医学问答，没有身份/自我介绍类样本，
导致 "你是谁" 被映射成 whoami 工具调用、"什么是文件" 输出退化文本。

方案：构造 ~140 条身份/闲聊样本 + 从原 SFT 集回放 4000 条工具样本防遗忘，
从 sft.pt 出发低学习率补训 2 epoch，存到 checkpoints/sft_v2.pt。

用法：
  ./.venv/Scripts/python.exe patch_sft.py
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from trainer.data import SFTDataset, build_chat, collate  # noqa: E402
from trainer.sft import evaluate  # noqa: E402
from trainer.utils import cosine_lr, describe_env, load_ckpt, make_optimizer, save_ckpt  # noqa: E402

ROOT = Path(__file__).resolve().parent

# 身份/闲聊样本：(问法, 回答)。覆盖同一意图的多种说法。
IDENTITY = [
    ("你是谁", "我是 Unknow，一个 37M 参数的小型命令行助手模型。我可以帮你查看目录、搜索文本、读写文件和执行简单命令。"),
    ("你叫什么名字", "我叫 Unknow，是专门为操作文件和命令行训练的小模型。"),
    ("介绍一下你自己", "我是 Unknow，一个超小型语言模型，擅长在项目目录里查文件、搜内容、做修改和跑命令。对话能力简单，但工具操作是强项。"),
    ("你是干什么的", "我是一个命令行助手，能列目录、搜文本、读写文件、跑 Python 片段。有什么想让我在项目里做的吗？"),
    ("你能做什么", "我能：列目录、看文件树、数行数、读文件、搜索文本、查找文件、复制/删除/追加/替换内容、跑 Python 代码、看磁盘和时间。"),
    ("你会什么", "我会一套文件和命令行工具：list_dir、tree、grep_text、find_ext、read_head、python_run 等等，可以在项目目录里直接帮你干活。"),
    ("你好", "你好！我是 Unknow，需要我帮你看点什么文件吗？"),
    ("在吗", "在的，我是 Unknow。想查什么文件或者跑什么命令，直接说。"),
    ("hi", "你好，我是 Unknow，可以帮你操作项目目录里的文件。"),
    ("谢谢", "不客气，还有需要随时说。"),
    ("再见", "好的，再见！"),
    ("什么是文件", "文件是存储在磁盘上的一段有名字的数据。在命令行里，文件可以用路径表示，比如 build/app.log，我可以帮你查看它的内容。"),
    ("什么是目录", "目录是用来组织文件的容器，里面可以放文件，也可以放子目录。我可以用 list_dir 或 tree 帮你看某个目录里有什么。"),
    ("什么是命令行", "命令行是用文本命令操作计算机的方式。我背后就跑在一个受限的 bash 沙箱里，可以安全地执行查看类命令。"),
]

MULTI = [
    [("你是谁", "我是 Unknow，一个 37M 的命令行助手小模型。"),
     ("你能帮我做什么", "我可以在项目目录里列目录、搜文本、读写文件、跑 Python 片段。")],
    [("你好", "你好！我是 Unknow。"),
     ("谢谢", "不客气！")],
]


def build_patch_items(tok) -> list[tuple[list[int], list[int]]]:
    rng = random.Random(7)
    items: list[tuple[list[int], list[int]]] = []
    # 单轮：每种问法做两遍，答案带一点随机变体（名字大小写/句尾）
    for q, a in IDENTITY:
        for _ in range(2):
            rec = {"msgs": [{"role": "user", "content": q},
                            {"role": "assistant", "content": a}]}
            it = build_chat(tok, rec["msgs"])
            if it:
                items.append(it)
    # 多轮
    for msgs in MULTI:
        for _ in range(2):
            it = build_chat(tok, [{"role": r, "content": c} for r, c in msgs])
            if it:
                items.append(it)
    # 回放：从原 SFT 集抽工具样本防遗忘
    blob = torch.load(ROOT / "data/tokenized/sft.pt", weights_only=False)
    tool_items = [it for it in blob["items"] if len(it[0]) > 40]
    items += rng.sample(tool_items, 4000)
    rng.shuffle(items)
    return items


def main() -> None:
    import argparse

    from transformers import PreTrainedTokenizerFast

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/sft.pt")
    ap.add_argument("--out", default="checkpoints/sft_v2.pt")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {describe_env()}")

    tok = PreTrainedTokenizerFast.from_pretrained(str(ROOT / "tokenizer/tokenizer_v3"))
    model, ck, cfg = load_ckpt(ROOT / a.ckpt)
    model.to(dev)

    items = build_patch_items(tok)
    n_val = 120
    val_items, train_items = items[:n_val], items[n_val:]
    print(f"补丁样本: 训练 {len(train_items)} / 验证 {n_val}")

    train_dl = DataLoader(SFTDataset(train_items), batch_size=8, shuffle=True,
                          generator=torch.Generator().manual_seed(0), collate_fn=collate)
    val_dl = DataLoader(SFTDataset(val_items), batch_size=8, shuffle=False, collate_fn=collate)

    steps = 2 * len(train_dl)
    opt = make_optimizer(model, lr=6e-5, weight_decay=0.01)

    def set_lr(lr: float) -> None:
        for g in opt.param_groups:
            g["lr"] = lr

    step = 0
    model.train()
    while step < steps:
        for x, y in train_dl:
            if step >= steps:
                break
            x, y = x.to(dev), y.to(dev)
            _, loss = model(x, y, loss_mask=(y != -100).long())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            set_lr(cosine_lr(step, steps, warmup=15, base_lr=6e-5, min_ratio=0.2))
            if step % 50 == 0:
                print(f"  step {step}/{steps} loss {loss.item():.3f} lr {opt.param_groups[0]['lr']:.2e}")
            step += 1

    vl = evaluate(model, val_dl, dev)
    print(f"补丁验证 loss: {vl:.3f}")
    save_ckpt(str(ROOT / a.out), model, None, step, cfg,
              val_loss=vl, note="sft + identity patch")
    print(f"已保存 {a.out}")


if __name__ == "__main__":
    main()
