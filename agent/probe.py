"""口语化探针：不看模板，只看模型在真实说法下的工具选择。

评测集是合成模板生成的，容易和训练集同源。这个探针用的说法是另写的，
覆盖同一批工具的不同表达，用来判断"换个说法还认不认得"。

    python -m agent.probe --ckpt checkpoints/sft.pt
    python -m agent.probe --ckpt checkpoints/sft.pt --show-args
"""
from __future__ import annotations

import argparse
import json

CASES: list[tuple[str, str, dict]] = [
    # (用户说法, 期望工具, 期望必填参数)
    ("这目录下都有啥", "list_dir", {"path": "."}),
    ("build 里面有些什么文件", "list_dir", {"path": "build"}),
    ("帮我看下 src 的目录结构", "tree", {"path": "src"}),
    ("build/app.log 有多少行", "count_lines", {"path": "build/app.log"}),
    ("logs/error.log 一共几行", "count_lines", {"path": "logs/error.log"}),
    ("把 build/app.log 开头几行打出来", "read_head", {"path": "build/app.log"}),
    ("我想看 docs/guide.md 的前 5 行", "read_head", {"path": "docs/guide.md"}),
    ("在项目里搜一下 TODO", "grep_text", {"pattern": "TODO"}),
    ("哪儿用到了 ERROR 这个词", "grep_text", {"pattern": "ERROR"}),
    ("找一下所有 .log 文件", "find_ext", {"ext": "log"}),
    ("哪些文件超过 2MB", "find_large", {"path": ".", "min_mb": 2}),
    ("把 tmp/x.bin 清空", "truncate_file", {"path": "tmp/x.bin"}),
    ("清空 logs/error.log 的内容", "truncate_file", {"path": "logs/error.log"}),
    ("把 build/notes.txt 里的 TODO 换成 DONE", "replace_text",
     {"path": "build/notes.txt", "old": "TODO", "new": "DONE"}),
    ("给 build/notes.txt 结尾加上 done", "append_file",
     {"path": "build/notes.txt", "text": "done"}),
    ("把 src/utils.py 复制到 docs/copy.md", "copy_file",
     {"src": "src/utils.py", "dst": "docs/copy.md"}),
    ("删掉 build/old.bak", "delete_file", {"path": "build/old.bak"}),
    ("看看 docs 占多少空间", "du_dir", {"path": "docs"}),
    ("现在几点了", "date_now", {}),
    ("磁盘还剩多少", "disk_free", {}),
    ("这些 CSV 按逗号切第 2 列", "cut_columns", {"sep": ",", "col": 2}),
    ("用 python 算一下 2 的 10 次方", "python_run", {"code": "print(2**10)"}),
    ("data/users.csv 排序一下", "sort_file", {"path": "data/users.csv"}),
    ("建个 docs/api 目录", "mkdir", {"path": "docs/api"}),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/sft.pt")
    ap.add_argument("--tokenizer", default="tokenizer/tokenizer_v3")
    ap.add_argument("--cwd", default="tools/_fixture")
    ap.add_argument("--show-args", action="store_true")
    a = ap.parse_args()

    import shutil
    import tempfile
    from pathlib import Path

    from agent.loop import Agent  # 延迟导入：要等 torch 起来

    # 探针里有删除/清空类指令，必须跑在副本上，别把夹具改了
    tmp = Path(tempfile.mkdtemp(prefix="probe_"))
    shutil.copytree(a.cwd, tmp, dirs_exist_ok=True)
    agent = Agent(a.ckpt, a.tokenizer, str(tmp), confirm=lambda *_: True,
                  verbose=False, temperature=0.0)

    name_ok = arg_ok = 0
    bad: list[str] = []
    for text, want_tool, want_args in CASES:
        try:
            agent.ids.clear()
            out = agent.step(text)
            last = agent.trace[-1] if getattr(agent, "trace", None) else None
            got_tool = last["tool"] if last else None
            got_args = last["args"] if last else {}
        except Exception as e:  # noqa: BLE001
            got_tool, got_args, out = None, {}, f"<异常 {e}>"
        req = {k: v for k, v in got_args.items() if k in want_args}
        hit_name = got_tool == want_tool
        hit_args = hit_name and req == want_args
        name_ok += hit_name
        arg_ok += hit_args
        mark = "✓" if hit_args else ("≈" if hit_name else "✗")
        line = f"{mark} {text[:24]:26s} 期望 {want_tool:13s} 实际 {str(got_tool):13s}"
        if a.show_args and not hit_args:
            line += f"\n     期望参数 {want_args} / 实际 {got_args}"
        print(line)
        if not hit_name:
            bad.append(text)
    n = len(CASES)
    print(f"\n工具选择 {name_ok}/{n} = {name_ok/n*100:.0f}% | 参数全对 {arg_ok}/{n} = {arg_ok/n*100:.0f}%")
    print(json.dumps({"tool": name_ok / n, "args": arg_ok / n, "ckpt": a.ckpt}, ensure_ascii=False))
    if bad:
        print("选错工具:", "、".join(bad))
    agent.close()
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
