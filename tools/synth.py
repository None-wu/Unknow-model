"""工具轨迹合成器：模板 × 真实执行 × 结果校验。

为什么不用大模型编：编出来的 stdout 是假的，模型会学到"命令和输出之间的虚假关联"。
这里每条轨迹都在临时沙箱里真跑一遍，拿真实 stdout/stderr 当标签，
再用规则把结果转成自然语言回答——数据自带 ground truth。

产出的三种样本：
  single  单步调用（含真实失败样本）
  chain   2~5 步串联，后一步参数依赖前一步输出
  refuse  越界请求，模型不出工具、直接拒绝（安全边界）
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.executor import BashWorker, Sandbox  # noqa: E402
from tools.schema import TOOL_MAP, TOOLS, render_command, Tool  # noqa: E402

DIRS = ["build", "src", "data", "logs", "docs", "tmp", "."]
FILES = [
    "build/app.log", "build/small.log", "build/notes.txt", "build/old.bak",
    "src/main.py", "src/utils.py", "src/config.json",
    "data/users.csv", "data/orders.csv", "data/empty.txt",
    "logs/access.log", "logs/error.log", "docs/guide.md", "docs/changelog.md",
    "README.md", "run.sh", "tmp/cache.bin", "data/events.jsonl",
]
import itertools
_FILE_WHEEL = itertools.cycle(FILES)
CSV_FILES = ["data/users.csv", "data/orders.csv"]
LOG_FILES = ["build/app.log", "logs/access.log", "logs/error.log"]
PATTERNS = ["TODO", "ERROR", "hello", "version", "api", "500", "def", "cache"]
MISSING = ["build/missing.log", "data/notfound.csv", "logs/old.log", "tmp/x.bin"]

PREFIXES = ["", "", "", "帮我", "麻烦", "请", "能不能", "我想要你"]
DISPLAY_PATH = {".": "当前目录"}
NEW_FILES = ["tmp/new.txt", "build/tmp.txt", "docs/draft.md", "tmp/cache2.bin", "logs/today.log"]

CHAT_SYSTEM_HINTS = {"role": "system", "content": "你可以调用命令行工具完成任务。"}


# ---------------------------------------------------------------------------
# 事件结构
# ---------------------------------------------------------------------------
@dataclass
class Step:
    name: str
    args: dict[str, Any]
    cmd: str = ""
    output: str = ""
    ok: bool = False


@dataclass
class Trajectory:
    intent: str
    steps: list[Step] = field(default_factory=list)
    answer: str = ""
    kind: str = "single"

    def to_record(self) -> dict:
        return {
            "kind": self.kind,
            "events": [
                {"t": "user", "text": self.intent},
                *[
                    {"t": "call", "name": s.name, "args": s.args,
                     "output": s.output, "ok": s.ok}
                    for s in self.steps
                ],
                {"t": "assistant", "text": self.answer},
            ],
            "meta": {"cmd": [s.cmd for s in self.steps]},
        }


def natural(tpl: str, **kw) -> str:
    return tpl.format(**kw)


# 同义改写表：模板只给出一种说法，模型换个词就认不出来。
# 这里在生成时随机替换动词，把"说法"的分布拉宽——比加模板便宜，且不改变语义。
SYNONYMS: list[tuple[str, list[str]]] = [
    ("查看", ["看看", "看一下", "瞅一眼", "瞧瞧"]),
    ("列出", ["列一下", "都列出来", "看看都有哪些"]),
    ("统计", ["数一下", "算一下", "数数"]),
    ("删除", ["删掉", "清掉", "去掉"]),
    ("复制", ["拷贝", "复制一份"]),
    ("移动", ["挪到", "移到"]),
    ("重命名", ["改名", "换个名字"]),
    ("搜索", ["搜一下", "查找", "找一下"]),
    ("读取", ["读一下", "看一下"]),
    ("创建", ["新建", "建一个"]),
    ("显示", ["打印出来", "输出一下"]),
    ("清空", ["清掉", "清零"]),
    ("开头", ["前面", "最前面", "头部"]),
    ("结尾", ["末尾", "最后面", "尾部"]),
    ("目录", ["文件夹"]),
    ("占多少", ["占了多大", "用掉多少"]),
]


def _paraphrase(rng: random.Random, text: str) -> str:
    """随机替换一处说法；改完不影响语义，只是让措辞更散。"""
    hits = [(w, subs) for w, subs in SYNONYMS if w in text]
    if not hits or rng.random() < 0.35:
        return text
    word, subs = rng.choice(hits)
    return text.replace(word, rng.choice(subs), 1)


def _phrase(rng: random.Random, base: str) -> str:
    return _paraphrase(rng, f"{rng.choice(PREFIXES)}{base}".strip())


def _clip(text: str, n: int = 160) -> str:
    text = text.strip()
    return text if len(text) <= n else text[:n] + "…"


# ---------------------------------------------------------------------------
# 结果 -> 自然语言（规则生成，保证不编造）
# ---------------------------------------------------------------------------
def answer_for(step: Step, tool: Tool) -> str:
    out = step.output.strip()
    if not step.ok:
        first = next((l for l in out.splitlines() if l.strip()), "命令执行失败")
        first = re.sub(r"^.*?: ", "", first)[:80]
        return f"执行失败了：{first}。请确认路径或参数是否正确。"

    if tool.name in ("list_dir", "tree"):
        items = [l for l in out.splitlines() if l.strip() and l.strip() not in (".", "..")]
        if not items:
            return "目录是空的。"
        shown = "、".join(items[:6])
        more = f"等 {len(items)} 项" if len(items) > 6 else f"共 {len(items)} 项"
        return f"目录里有 {shown}，{more}。"
    if tool.name in ("grep_text", "find_name", "find_ext", "find_large", "find_empty", "top_files"):
        lines = [l for l in out.splitlines() if l.strip()]
        if not lines:
            return "没有找到匹配的内容。"
        return f"找到 {len(lines)} 条结果，例如 {_clip('；'.join(lines[:3]), 120)}。"
    if tool.name in ("read_head", "read_tail", "read_range", "cat_file"):
        return f"内容是：{_clip(out, 140)}"
    if tool.name in ("count_lines", "count_files", "grep_count"):
        num = re.search(r"\d+", out)
        return f"共 {num.group(0) if num else '未知'} 项。"
    if tool.name == "word_count":
        parts = out.split()
        if len(parts) >= 3:
            return f"{parts[2]} 字节，{parts[1]} 个词，{parts[0]} 行。"
        return f"统计结果是：{_clip(out, 80)}"
    if tool.name == "du_dir":
        return f"占用空间 {out.split()[0] if out else '未知'}。"
    if tool.name in ("delete_file", "delete_dir"):
        return f"已删除 {step.args.get('path')}。"
    if tool.name in ("mkdir",):
        return f"已创建目录 {step.args.get('path')}。"
    if tool.name in ("touch",):
        return f"已创建文件 {step.args.get('path')}。"
    if tool.name in ("write_file", "append_file"):
        return f"已写入 {step.args.get('path')}。"
    if tool.name in ("copy_file", "move_file"):
        dst = step.args.get("dst", "")
        verb = "复制" if tool.name == "copy_file" else "移动"
        if "." in Path(dst).name and "/" in dst or dst.endswith((".bak", ".log", ".txt", ".md")):
            return f"已把 {step.args.get('src')} {verb}到 {dst}。"
        return f"已把 {step.args.get('src')} {verb}到 {dst}/ 目录下。"
    if tool.name == "replace_text":
        return f"已把 {step.args.get('path')} 中的 {step.args.get('old')} 替换为 {step.args.get('new')}。"
    if tool.name == "disk_free":
        for line in out.splitlines():
            parts = line.split()
            if len(parts) >= 4 and parts[-2].endswith("%"):
                return f"磁盘剩余 {parts[3]}，已用 {parts[-2]}。"
        return f"磁盘情况：{_clip(out, 100)}"
    if tool.name == "count_files":
        num = re.search(r"\d+", out)
        return f"目录下共有 {num.group(0) if num else '未知'} 个文件。"
    if tool.name == "truncate_file":
        return f"已清空 {step.args.get('path')} 的内容，文件本身保留。"
    if tool.name in ("stat_file", "file_type", "diff_files", "sort_file", "uniq_count",
                     "cut_columns", "python_run"):
        return f"结果如下：{_clip(out, 140)}"
    if tool.name in ("disk_free", "date_now", "whoami", "list_process", "which_cmd", "env_get"):
        return f"当前结果是：{_clip(out, 120)}"
    return f"执行完成：{_clip(out, 120)}"


# ---------------------------------------------------------------------------
# 单步生成
# ---------------------------------------------------------------------------
def pick_args(rng: random.Random, tool: Tool, force_missing: bool = False) -> dict[str, Any]:
    args: dict[str, Any] = {}
    for a in tool.args:
        if not a.required and rng.random() < 0.4:
            continue
        if a.type == "path":
            fileish = a.name in ("path",) and tool.name in (
                "read_head", "read_tail", "read_range", "cat_file", "stat_file", "file_type",
                "count_lines", "word_count", "sort_file", "uniq_count", "diff_files",
                "truncate_file", "copy_file", "move_file", "delete_file", "replace_text",
                "cut_columns", "grep_count", "append_file", "write_file",
            )
            # 轮转取文件，而不是随机取：随机采样下每个工具看到的扩展名分布会偏，
            # 模型就会把"清空 .bin"错配成 delete/move（truncate 从没见过 .bin）。
            pool = FILES if fileish else DIRS
            if tool.name == "grep_text":
                pool = ["."] + DIRS + FILES
            if tool.name in ("touch", "write_file", "append_file"):
                pool = NEW_FILES
            if tool.name == "mkdir":
                pool = ["tmp/out", "build/logs", "docs/api", "data/raw"]
            if tool.name in ("copy_file", "move_file"):
                pool = FILES if a.name == "src" else ["tmp/moved.bak", "build/renamed.log", "docs/copy.md"]
            if tool.name in ("delete_dir", "du_dir", "count_files"):
                pool = ["tmp", "build", "docs"]
            if tool.name == "find_large" or tool.name == "top_files":
                pool = ["."] + DIRS
            v = next(_FILE_WHEEL) if pool is FILES else rng.choice(pool)
            if force_missing and a.required:
                v = rng.choice(MISSING)
            args[a.name] = v
        elif a.type == "pattern":
            if a.name == "old":
                args[a.name] = rng.choice(["TODO", "ERROR", "hello"])
            elif a.name == "pattern":
                # 按文件名找 与 按内容搜 是两种语义，参数池必须分开：
                # 以前 find_name 也吃 TODO/ERROR，模型就学会了 `find . -name TODO`。
                if "find" in tool.name:
                    args[a.name] = rng.choice(["*.py", "*.log", "*.txt", "*.md", "*.csv",
                                               "*.bin", "main*", "config*", "*.json"])
                else:
                    args[a.name] = rng.choice(PATTERNS)
        elif a.type == "text":
            if a.name == "new":
                args[a.name] = rng.choice(["DONE", "FIXME", "ok"])
            elif a.name == "text":
                args[a.name] = rng.choice(["hello world", "192.168.1.1", "debug=true", "1,2,3"])
        elif a.type == "cmd":
            args[a.name] = rng.choice([
                "print(sum(range(10)))",
                "print(len('hello'))",
                "print(2**10)",
                "print(round(3.14159, 2))",
                "print(sorted([3,1,2]))",
                "print('ok' if 7 % 2 else 'even')",
            ])
        elif a.type == "int":
            lo = a.minimum or 1
            hi = min(a.maximum or 10, 20)
            if a.name in ("start",):
                args[a.name] = rng.choice([1, 2, 5])
            elif a.name in ("end",):
                args[a.name] = rng.choice([10, 15, 20])
            elif a.name == "min_mb":
                args[a.name] = rng.choice([1, 2, 3])
            elif a.name == "col":
                args[a.name] = rng.choice([1, 2, 3])
            else:
                args[a.name] = rng.randint(lo, min(hi, 10))
        elif a.type == "choice":
            args[a.name] = rng.choice(a.choices or ["x"])
        elif a.type == "bool":
            args[a.name] = rng.random() < 0.5
    return args


def make_intent(rng: random.Random, tool: Tool, args: dict[str, Any]) -> str:
    tpl = rng.choice(tool.intents)
    slots = {k: (DISPLAY_PATH.get(v, v) if isinstance(v, str) else v) for k, v in args.items()}
    slots.setdefault("lines", 10)
    slots.setdefault("n", 5)
    slots.setdefault("name", "python")
    slots.setdefault("ext", "py")
    try:
        base = tpl.format(**slots)
    except (KeyError, IndexError):
        base = f"{tool.desc}，参数是 {args}"
    return _phrase(rng, base)


def single_trajectory(rng: random.Random, sb: Sandbox, tool: Tool) -> Trajectory | None:
    force_missing = rng.random() < 0.12
    args = pick_args(rng, tool, force_missing=force_missing)
    try:
        cmd = render_command(tool, args)
    except (KeyError, ValueError):
        return None
    res = sb.run(cmd)
    step = Step(tool.name, args, cmd, res.output, res.ok)
    intent = make_intent(rng, tool, args)
    return Trajectory(intent, [step], answer_for(step, tool), "single")


# ---------------------------------------------------------------------------
# 多步链：后一步参数来自前一步输出
# ---------------------------------------------------------------------------
def chain_bigfile_delete(rng: random.Random, sb: Sandbox) -> Trajectory | None:
    s1 = Step("find_large", {"path": ".", "min_mb": 2})
    s1.cmd = render_command(TOOL_MAP["find_large"], s1.args)
    r1 = sb.run(s1.cmd)
    s1.output, s1.ok = r1.output, r1.ok
    files = [l.strip().removeprefix("./") for l in r1.output.splitlines() if l.strip()]
    if not files:
        return None
    target = files[0]
    s2 = Step("stat_file", {"path": target})
    s2.cmd = render_command(TOOL_MAP["stat_file"], s2.args)
    r2 = sb.run(s2.cmd)
    s2.output, s2.ok = r2.output, r2.ok
    s3 = Step("delete_file", {"path": target})
    s3.cmd = render_command(TOOL_MAP["delete_file"], s3.args)
    r3 = sb.run(s3.cmd)
    s3.output, s3.ok = r3.output, r3.ok
    if not (s1.ok and s2.ok and s3.ok):
        return None
    size = r2.output.split()[1] if len(r2.output.split()) > 1 else ""
    intent = _phrase(rng, rng.choice([
        "找出当前目录下超过 2MB 的大文件，确认后删掉它",
        "帮我清理一下大于 2MB 的垃圾文件",
        "磁盘不够了，把那个大文件删掉",
    ]))
    ans = f"最大的是 {target}，{size} 字节，已经删除。"
    return Trajectory(intent, [s1, s2, s3], ans, "chain")


def chain_search_read(rng: random.Random, sb: Sandbox) -> Trajectory | None:
    pat = rng.choice(["TODO", "ERROR", "api", "def"])
    s1 = Step("grep_text", {"pattern": pat, "path": "."})
    s1.cmd = render_command(TOOL_MAP["grep_text"], s1.args)
    r1 = sb.run(s1.cmd)
    s1.output, s1.ok = r1.output, r1.ok
    lines = [l for l in r1.output.splitlines() if l.strip()]
    if not lines:
        return None
    f = lines[0].split(":")[0].replace("./", "")
    s2 = Step("read_head", {"path": f, "lines": 5})
    s2.cmd = render_command(TOOL_MAP["read_head"], s2.args)
    r2 = sb.run(s2.cmd)
    s2.output, s2.ok = r2.output, r2.ok
    intent = _phrase(rng, f"在项目里搜索 {pat}，找到后看看那个文件开头写了什么")
    ans = f"在 {len(lines)} 处找到 {pat}，{f} 的内容是：{_clip(r2.output, 100)}"
    return Trajectory(intent, [s1, s2], ans, "chain")


def chain_mkdir_move(rng: random.Random, sb: Sandbox) -> Trajectory | None:
    tag = rng.choice(["archive", "backup", "old", "trash"])
    s1 = Step("mkdir", {"path": tag})
    s1.cmd = render_command(TOOL_MAP["mkdir"], s1.args)
    r1 = sb.run(s1.cmd)
    s1.output, s1.ok = r1.output, r1.ok
    src = rng.choice(["build/old.bak", "build/small.log", "docs/changelog.md"])
    s2 = Step("move_file", {"src": src, "dst": f"{tag}/{Path(src).name}"})
    s2.cmd = render_command(TOOL_MAP["move_file"], s2.args)
    r2 = sb.run(s2.cmd)
    s2.output, s2.ok = r2.output, r2.ok
    s3 = Step("list_dir", {"path": tag})
    s3.cmd = render_command(TOOL_MAP["list_dir"], s3.args)
    r3 = sb.run(s3.cmd)
    s3.output, s3.ok = r3.output, r3.ok
    intent = _phrase(rng, f"新建一个 {tag} 目录，把 {src} 挪进去，然后确认一下")
    ans = f"已把 {src} 移动到 {tag}/，目录现在有 {len([l for l in r3.output.splitlines() if l.strip() and l.strip() not in ('.', '..')])} 个文件。"
    return Trajectory(intent, [s1, s2, s3], ans, "chain")


def chain_log_count(rng: random.Random, sb: Sandbox) -> Trajectory | None:
    f = rng.choice(LOG_FILES)
    s1 = Step("read_tail", {"path": f, "lines": 5})
    s1.cmd = render_command(TOOL_MAP["read_tail"], s1.args)
    r1 = sb.run(s1.cmd)
    s1.output, s1.ok = r1.output, r1.ok
    s2 = Step("grep_count", {"pattern": "ERROR", "path": f})
    s2.cmd = render_command(TOOL_MAP["grep_count"], s2.args)
    r2 = sb.run(s2.cmd)
    s2.output, s2.ok = r2.output, r2.ok
    num = re.search(r"\d+", r2.output)
    intent = _phrase(rng, f"看看 {f} 最近的日志，顺便统计有多少条 ERROR")
    ans = f"最近几行是：{_clip(r1.output, 80)}；ERROR 一共出现 {num.group(0) if num else 0} 次。"
    return Trajectory(intent, [s1, s2], ans, "chain")


def chain_csv_col(rng: random.Random, sb: Sandbox) -> Trajectory | None:
    f = rng.choice(CSV_FILES)
    s1 = Step("read_head", {"path": f, "lines": 3})
    s1.cmd = render_command(TOOL_MAP["read_head"], s1.args)
    r1 = sb.run(s1.cmd)
    s1.output, s1.ok = r1.output, r1.ok
    s2 = Step("cut_columns", {"path": f, "sep": ",", "col": 2})
    s2.cmd = render_command(TOOL_MAP["cut_columns"], s2.args)
    r2 = sb.run(s2.cmd)
    s2.output, s2.ok = r2.output, r2.ok
    intent = _phrase(rng, f"把 {f} 的第二列取出来，先告诉我表头是什么")
    ans = f"表头是 {r1.output.splitlines()[0] if r1.output else '未知'}，第二列内容为：{_clip(r2.output, 80)}"
    return Trajectory(intent, [s1, s2], ans, "chain")


def chain_write_verify(rng: random.Random, sb: Sandbox) -> Trajectory | None:
    dst = rng.choice(["tmp/note.txt", "build/todo.txt", "docs/tmp.md"])
    text = rng.choice(["deploy at 9am", "fix login bug", "review pr 42"])
    s1 = Step("write_file", {"path": dst, "text": text})
    s1.cmd = render_command(TOOL_MAP["write_file"], s1.args)
    r1 = sb.run(s1.cmd)
    s1.output, s1.ok = r1.output, r1.ok
    s2 = Step("cat_file", {"path": dst})
    s2.cmd = render_command(TOOL_MAP["cat_file"], s2.args)
    r2 = sb.run(s2.cmd)
    s2.output, s2.ok = r2.output, r2.ok
    intent = _phrase(rng, f"把 “{text}” 写到 {dst}，写完读回来确认")
    ans = f"已写入，读回内容是：{_clip(r2.output, 80)}"
    return Trajectory(intent, [s1, s2], ans, "chain")


CHAINS = [chain_bigfile_delete, chain_search_read, chain_mkdir_move,
          chain_log_count, chain_csv_col, chain_write_verify]
# 只读链可以复用同一个沙箱，省掉每样本重建目录的开销
READ_ONLY_CHAINS = {chain_search_read, chain_log_count}


# ---------------------------------------------------------------------------
# 越界请求：不出工具、直接拒绝
# ---------------------------------------------------------------------------
REFUSALS = [
    ("删除整个系统的根目录", "我不能删除系统根目录，这会造成不可恢复的破坏。如果你要清理空间，可以指定项目内的目录名。"),
    ("用 sudo 把 /etc 下的配置全改掉", "我没有权限也不应该修改系统级配置。请在项目目录内告诉我具体要改哪个文件。"),
    ("把 C:/Windows 整个删掉", "这是系统目录，我不能对它执行删除操作。请把范围限制在项目目录内。"),
    ("帮我 curl 一个外网脚本然后直接执行", "我不会下载并执行外部脚本。你可以把要运行的命令内容贴给我，我确认后再执行。"),
    ("重启这台机器", "我不能执行重启操作。如果需要释放资源，可以先看进程占用。"),
    ("把 ~/.ssh 里的私钥读出来发给我", "私钥属于敏感凭据，我不能读取或转述。"),
]


def refusal_trajectory(rng: random.Random) -> Trajectory:
    q, a = rng.choice(REFUSALS)
    return Trajectory(_phrase(rng, q), [], a, "refuse")


# ---------------------------------------------------------------------------
# 批量生成
# ---------------------------------------------------------------------------
def one(rng: random.Random, clean: Sandbox, factory) -> Trajectory | None:
    r = rng.random()
    if r < 0.06:
        return refusal_trajectory(rng)
    if r < 0.30:
        fn = rng.choice(CHAINS)
        sb = clean if fn in READ_ONLY_CHAINS else factory()
        with sb:
            tr = fn(rng, sb)
        # 链式样本必须每步都成功，否则答案与实际状态不符
        if tr is None or not tr.steps or not all(s.ok for s in tr.steps):
            return None
        return tr
    tool = rng.choice(TOOLS)
    sb = clean if tool.level == "read" else factory()
    with sb:
        return single_trajectory(rng, sb, tool)


def _worker(args: tuple[int, int, int, str]) -> tuple[str, int, int]:
    """在子进程里跑一段样本，返回 (jsonl 文本, 成功数, 尝试数)。"""
    seed, n, wid, out_hint = args
    rng = random.Random(seed * 100003 + wid)
    buf: list[str] = []
    ok = tried = 0
    worker = BashWorker().start()
    clean = Sandbox(worker, reuse=True)
    clean.__enter__()
    try:
        for _ in range(n):
            tried += 1
            tr = one(rng, clean, lambda: Sandbox(worker))
            if tr is None or (not tr.steps and tr.kind != "refuse"):
                continue
            if tr.steps and not any(s.ok for s in tr.steps):
                continue  # 全部失败且不是有意构造的失败样本
            buf.append(json.dumps(tr.to_record(), ensure_ascii=False))
            ok += 1
    finally:
        clean.destroy()
        worker.close()
    return "\n".join(buf), ok, tried


def main() -> None:
    ap = argparse.ArgumentParser(description="合成工具调用 SFT 轨迹")
    ap.add_argument("--n", type=int, default=20000, help="目标样本数")
    ap.add_argument("--workers", type=int, default=0, help="并行进程数（0=CPU 核数-1）")
    ap.add_argument("--out", type=str, default="data/tool_sft.jsonl")
    ap.add_argument("--seed", type=int, default=20261103)
    ap.add_argument("--shard", type=int, default=500, help="每个子任务的样本数")
    a = ap.parse_args()

    workers = a.workers or max(1, (os.cpu_count() or 2) - 1)
    shards = max(workers, a.n // a.shard + (1 if a.n % a.shard else 0))
    per = a.n // shards + 1
    jobs = [(a.seed, per, i, a.out) for i in range(shards)]

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    total_ok = total_tried = 0
    with open(out, "w", encoding="utf-8") as f, ctx.Pool(workers) as pool:
        for i, (text, ok, tried) in enumerate(pool.imap_unordered(_worker, jobs), 1):
            if text:
                f.write(text + "\n")
                f.flush()
            total_ok += ok
            total_tried += tried
            print(f"  分片 {i}/{len(jobs)} 完成：+{ok} 条（累计 {total_ok}）", flush=True)
    print(f"\n完成：{total_ok} 条 / 尝试 {total_tried} 条，通过率 {total_ok/max(1,total_tried)*100:.1f}%")
    print(f"输出：{out.resolve()}")


if __name__ == "__main__":
    main()
