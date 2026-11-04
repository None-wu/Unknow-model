"""评测：工具选择准确率 / 参数完全匹配 / 调用体合法性 / 可执行率。

四个指标分别回答四个问题：
  legal    <- 模型吐出来的东西能不能解析成调用（约束解码应接近 100%）
  tool     <- 选对工具了吗
  args     <- 参数一字不差吗（路径、枚举、整数都要对）
  exec_ok  <- 在沙箱里真跑一遍，能跑通吗（语义可用性）

用法：
  python -m agent.eval --ckpt checkpoints/sft.pt --data data/tool_eval.jsonl --n 100
  python -m agent.eval --ckpt checkpoints/sft.pt --data data/tool_eval.jsonl --n 100 --free
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import torch

from agent.decoder import ConstrainedDecoder
from model.model import MiniGPT
from tools.executor import BashWorker, Sandbox, candidates_for, fixture_paths
from tools.schema import (A_TOKEN, END_TOKEN, TOOL_MAP, U_TOKEN,
                          render_command, validate_args)

ROOT = Path(__file__).resolve().parent.parent


def free_parse(text: str) -> tuple[str, dict] | None:
    """不约束时的宽松解析：尽量从文本里抠出 name/args。"""
    s = text.replace("<|tool_call|>", "").replace(END_TOKEN, "")
    start, end = s.find("{"), s.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(s[start:end + 1])
    except json.JSONDecodeError:
        return None
    name = obj.get("name") or obj.get("tool")
    args = obj.get("args") or obj.get("arguments") or {}
    if not isinstance(name, str) or name not in TOOL_MAP or not isinstance(args, dict):
        return None
    return name, args


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/sft.pt")
    ap.add_argument("--tokenizer", default="tokenizer/tokenizer_v3")
    ap.add_argument("--data", default="data/tool_eval.jsonl")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--constrained", action="store_true",
                    help="强制语法约束解码（默认自由生成，与 Agent 运行时策略一致）")
    ap.add_argument("--dump", default=None, help="错例写到这个文件")
    a = ap.parse_args()

    from transformers import PreTrainedTokenizerFast
    from trainer.utils import load_ckpt

    tok = PreTrainedTokenizerFast.from_pretrained(str(ROOT / a.tokenizer))
    model, _, cfg = load_ckpt(ROOT / a.ckpt)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev).eval()

    end_id = tok.convert_tokens_to_ids(END_TOKEN)
    a_id = tok.convert_tokens_to_ids(A_TOKEN)

    rows = [json.loads(l) for l in open(ROOT / a.data, encoding="utf-8")]
    rows = [r for r in rows if r.get("kind") == "single" and r["events"][0]["t"] == "user"]
    random.Random(0).shuffle(rows)
    rows = rows[: a.n]

    called = legal = tool_ok = args_ok = exec_ok = 0
    errors: list[dict] = []
    worker = BashWorker().start()
    FIXTURE = ROOT / 'tools/_fixture'

    for r in rows:
        gold = next((e for e in r["events"] if e["t"] == "call"), None)
        if not gold:
            continue
        prompt = f"{U_TOKEN}{r['events'][0]['text']}{END_TOKEN}{A_TOKEN}"
        x = torch.tensor([tok(prompt, add_special_tokens=False)["input_ids"]],
                         dtype=torch.long, device=dev)

        # 路径候选随每条意图变化：环境清单 + 话里点名的路径
        cands = (candidates_for(FIXTURE, r['events'][0]['text'])
                 if FIXTURE.exists() else fixture_paths())
        dec = ConstrainedDecoder(tok, path_candidates=cands, path_root=str(FIXTURE))
        if not a.constrained:
            out = model.generate(x, max_new_tokens=160, temperature=max(a.temp, 0.7),
                                 top_k=40, stop_ids={end_id})
        else:
            out = model.generate(x, max_new_tokens=200, temperature=a.temp, top_k=40,
                                 stop_ids={end_id}, logits_filter=dec, on_token=dec.observe)
        text = tok.decode(out[0].tolist(), skip_special_tokens=False)

        attempted = dec.saw_call or ("<|tool_call|>" in text)
        call = dec.call if (dec.saw_call and dec.call) else free_parse(text)
        if attempted:
            called += 1
        if call:
            legal += 1
            name, args = call
            tool = TOOL_MAP[name]
            req = {x.name for x in tool.args if x.required}
            gold_args = {k: v for k, v in gold["args"].items() if k in req}
            # 只比必填参数：模型多给一个可选参数（如 read_head 的 lines）不算错，
            # 金标本来就只含必填项，直接比较会把正确调用误判成错误。
            got_req = {k: v for k, v in args.items() if k in req}
            if name == gold["name"]:
                tool_ok += 1
                if got_req == gold_args:
                    args_ok += 1
                    try:
                        cmd = render_command(tool, args)
                        with Sandbox(worker) as sb:
                            if sb.run(cmd).ok:
                                exec_ok += 1
                    except Exception:
                        pass
            if name != gold["name"] or got_req != gold_args:
                errors.append({"intent": r["events"][0]["text"],
                               "gold": {"name": gold["name"], "args": gold_args},
                               "got": {"name": name, "args": got_req}})
        else:
            errors.append({"intent": r["events"][0]["text"], "gold": gold, "got": text[:200]})

    n = len(rows)
    print(f"\n样本 {n} | 模式 {'约束解码' if a.constrained else '自由生成'}")
    print(f"  出调用比例   : {called/n*100:5.1f}%")
    print(f"  已调用中合法 : {legal/max(1, called)*100:5.1f}%")
    print(f"  工具选择准确 : {tool_ok/n*100:5.1f}%")
    print(f"  参数完全匹配 : {args_ok/n*100:5.1f}%")
    print(f"  沙箱可执行   : {exec_ok/n*100:5.1f}%")

    if a.dump and errors:
        Path(a.dump).write_text(
            "\n".join(json.dumps(e, ensure_ascii=False) for e in errors), encoding="utf-8")
        print(f"  错例 {len(errors)} 条 -> {a.dump}")
    worker.close()


if __name__ == "__main__":
    main()
