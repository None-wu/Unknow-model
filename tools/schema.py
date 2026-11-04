"""工具 DSL 与对话协议定义。

设计原则（针对 40M 级小模型）：
1. 封闭集合：工具表固定，不允许模型自由发明命令；
2. 参数强类型：每个参数有类型和候选集，配合约束解码可做到 100% 合法；
3. 命令可回放：结构化调用 -> POSIX shell 命令，便于真实执行校验；
4. 协议用独立 special token 包裹，让模型把"说话"和"发命令"分成两种模式。

协议格式：
    <|user|>把 build 目录里的日志删掉<|end|>
    <|assistant|><|tool_call|>{"name":"find_large","args":{"path":"build","min_mb":10}}<|end|>
    <|tool|><|tool_result|>build/a.log\nbuild/b.log<|end|>
    <|assistant|><|tool_call|>{"name":"delete_file","args":{"path":"build/a.log"}}<|end|>
    <|tool|><|tool_result|>deleted build/a.log<|end|>
    <|assistant|>已删除 2 个超过 10MB 的日志文件。<|end|>
"""
from __future__ import annotations

import json
import shlex
from dataclasses import dataclass, field
from typing import Any, Literal

# ---------------------------------------------------------------------------
# 对话协议
# ---------------------------------------------------------------------------
BOS = "<s>"
EOS = "</s>"
PAD = "<pad>"
UNK = "<unk>"

U_TOKEN = "<|user|>"
A_TOKEN = "<|assistant|>"
T_TOKEN = "<|tool|>"
CALL_TOKEN = "<|tool_call|>"
RESULT_TOKEN = "<|tool_result|>"
END_TOKEN = "<|end|>"
THINK_TOKEN = "<|think|>"

PROTOCOL_TOKENS = [U_TOKEN, A_TOKEN, T_TOKEN, CALL_TOKEN, RESULT_TOKEN, END_TOKEN, THINK_TOKEN]
SPECIAL_TOKENS = [BOS, EOS, UNK, PAD] + PROTOCOL_TOKENS

ArgType = Literal["path", "text", "int", "bool", "choice", "pattern", "cmd"]


@dataclass
class Arg:
    name: str
    type: ArgType
    desc: str
    required: bool = True
    choices: list[str] | None = None
    minimum: int | None = None
    maximum: int | None = None


@dataclass
class Tool:
    name: str
    desc: str
    args: list[Arg]
    template: str
    intents: list[str]
    level: Literal["read", "write", "exec"] = "read"
    # 结果是否适合作为下一步的输入（文件列表类工具）
    emits_paths: bool = False
    tags: list[str] = field(default_factory=list)


TOOLS: list[Tool] = [
    # ---------------- 目录与文件浏览 ----------------
    Tool("list_dir", "列出目录下的文件和子目录", [Arg("path", "path", "目标目录")],
         "ls -1 -a {path}", ["看看 {path} 里有什么", "列出 {path} 的内容", "{path} 目录下有哪些文件"],
         emits_paths=True, tags=["dir"]),
    Tool("tree", "以树形展示目录结构（最多两层）", [Arg("path", "path", "目标目录")],
         "find {path} -maxdepth 2 | sort", ["用树形展示 {path} 的结构", "把 {path} 的目录结构列出来"],
         emits_paths=True, tags=["dir"]),
    Tool("stat_file", "查看文件的元信息（大小、权限、修改时间）",
         [Arg("path", "path", "目标文件")],
         "stat -c '%n %s bytes %y %A' {path}", ["{path} 这个文件多大", "查看 {path} 的详细信息", "看一下 {path} 的修改时间和大小"],
         tags=["file"]),
    Tool("file_type", "判断文件类型", [Arg("path", "path", "目标文件")],
         "file {path}", ["{path} 是什么类型的文件", "判断一下 {path} 的格式"], tags=["file"]),
    Tool("count_lines", "统计文件行数", [Arg("path", "path", "目标文件")],
         "wc -l < {path}", ["{path} 有多少行", "数一下 {path} 的行数"], tags=["file"]),
    Tool("du_dir", "统计目录占用空间", [Arg("path", "path", "目标目录")],
         "du -sh {path}", ["{path} 占了多大空间", "看看 {path} 总共占多少磁盘"], tags=["dir"]),

    # ---------------- 读取内容 ----------------
    Tool("read_head", "读取文件开头若干行",
         [Arg("path", "path", "目标文件"), Arg("lines", "int", "行数", required=False, minimum=1, maximum=50)],
         "head -n {lines} {path}", ["看看 {path} 开头写了什么", "读一下 {path} 的前几行", "打印 {path} 的开头"],
         tags=["read"]),
    Tool("read_tail", "读取文件末尾若干行",
         [Arg("path", "path", "目标文件"), Arg("lines", "int", "行数", required=False, minimum=1, maximum=50)],
         "tail -n {lines} {path}", ["看看 {path} 的最后几行", "把 {path} 结尾打印出来", "{path} 最新的日志是什么"],
         tags=["read"]),
    Tool("read_range", "读取文件指定行区间",
         [Arg("path", "path", "目标文件"), Arg("start", "int", "起始行", minimum=1, maximum=500),
          Arg("end", "int", "结束行", minimum=1, maximum=500)],
         "sed -n '{start},{end}p' {path}", ["读 {path} 第 {start} 到 {end} 行", "把 {path} 的 {start}-{end} 行打印出来"],
         tags=["read"]),
    Tool("cat_file", "输出整个文件内容（限小文件）", [Arg("path", "path", "目标文件")],
         "cat {path}", ["把 {path} 的内容全部打印出来", "看看 {path} 里写了什么"], tags=["read"]),

    # ---------------- 搜索 ----------------
    Tool("grep_text", "在文件中搜索固定字符串",
         [Arg("pattern", "pattern", "要搜索的字符串"), Arg("path", "path", "搜索范围")],
         "grep -rn -- {pattern} {path}", ["在 {path} 里找 {pattern}", "搜索 {path} 中包含 {pattern} 的地方",
                                          "哪里用到了 {pattern}"],
         emits_paths=True, tags=["search"]),
    Tool("grep_count", "统计关键词出现次数",
         [Arg("pattern", "pattern", "关键词"), Arg("path", "path", "目标文件")],
         "grep -c -- {pattern} {path}", ["{path} 里 {pattern} 出现了几次", "数一下 {path} 中 {pattern} 的数量"],
         tags=["search"]),
    Tool("find_name", "按文件名查找",
         [Arg("pattern", "pattern", "文件名模式"), Arg("path", "path", "搜索目录")],
         "find {path} -name {pattern}", ["找出 {path} 下所有名为 {pattern} 的文件",
                                          "在 {path} 里搜 {pattern} 这个文件"],
         emits_paths=True, tags=["search"]),
    Tool("find_ext", "按扩展名查找",
         [Arg("ext", "choice", "扩展名", choices=["log", "py", "txt", "json", "md", "csv", "sh"]),
          Arg("path", "path", "搜索目录")],
         "find {path} -name '*.{ext}'", ["找出 {path} 下所有 .{ext} 文件", "{path} 里有哪些 {ext} 文件"],
         emits_paths=True, tags=["search"]),
    Tool("find_large", "查找大于指定体积的文件",
         [Arg("path", "path", "搜索目录"), Arg("min_mb", "int", "体积下限 MB", minimum=1, maximum=1024)],
         "find {path} -type f -size +{min_mb}M", ["找出 {path} 下大于 {min_mb}MB 的文件",
                                                  "{path} 里有哪些超过 {min_mb}M 的大文件"],
         emits_paths=True, tags=["search"]),
    Tool("find_empty", "查找空文件或空目录", [Arg("path", "path", "搜索目录")],
         "find {path} -empty", ["找出 {path} 下的空文件", "{path} 里哪些文件是空的"],
         emits_paths=True, tags=["search"]),

    # ---------------- 写入与修改 ----------------
    Tool("mkdir", "创建目录", [Arg("path", "path", "目录路径")],
         "mkdir -p {path}", ["创建目录 {path}", "新建一个 {path} 文件夹", "帮我建个 {path}"],
         level="write", tags=["write"]),
    Tool("touch", "创建空文件或更新时间戳", [Arg("path", "path", "文件路径")],
         "touch {path}", ["新建一个空文件 {path}", "创建 {path}"], level="write", tags=["write"]),
    Tool("write_file", "把内容写入文件（覆盖）",
         [Arg("path", "path", "文件路径"), Arg("text", "text", "写入内容")],
         "printf '%s' {text} > {path}", ["把 {text} 写进 {path}", "用 {text} 覆盖 {path} 的内容",
                                          "{path} 改成 {text}"],
         level="write", tags=["write"]),
    Tool("append_file", "在文件末尾追加内容",
         [Arg("path", "path", "文件路径"), Arg("text", "text", "追加内容")],
         "printf '%s\\n' {text} >> {path}", ["给 {path} 末尾加上 {text}", "往 {path} 追加一行 {text}"],
         level="write", tags=["write"]),
    Tool("copy_file", "复制文件",
         [Arg("src", "path", "源文件"), Arg("dst", "path", "目标路径")],
         "cp {src} {dst}", ["把 {src} 复制到 {dst}", "备份 {src} 到 {dst}"], level="write", tags=["write"]),
    Tool("move_file", "移动或重命名文件",
         [Arg("src", "path", "源文件"), Arg("dst", "path", "目标路径")],
         "mv {src} {dst}", ["把 {src} 移到 {dst}", "把 {src} 重命名为 {dst}"], level="write", tags=["write"]),
    Tool("delete_file", "删除文件", [Arg("path", "path", "目标文件")],
         "rm -f {path}", ["删掉 {path}", "把 {path} 这个文件删了", "清理掉 {path}"],
         level="write", tags=["write", "danger"]),
    Tool("delete_dir", "递归删除目录", [Arg("path", "path", "目标目录")],
         "rm -rf {path}", ["把整个 {path} 目录删掉", "删除 {path} 文件夹及其内容"],
         level="write", tags=["write", "danger"]),
    Tool("truncate_file", "清空文件但保留文件本身", [Arg("path", "path", "目标文件")],
         ": > {path}", ["清空 {path} 的内容", "把 {path} 里的内容清掉但保留文件"],
         level="write", tags=["write"]),
    Tool("replace_text", "替换文件中的字符串",
         [Arg("path", "path", "目标文件"), Arg("old", "pattern", "被替换的文本"),
          Arg("new", "text", "替换后的文本")],
         "sed -i 's/{old}/{new}/g' {path}", ["把 {path} 里的 {old} 全部换成 {new}",
                                              "{path} 中的 {old} 改成 {new}"],
         level="write", tags=["write"]),

    # ---------------- 文本处理 ----------------
    Tool("sort_file", "对文件按行排序", [Arg("path", "path", "目标文件")],
         "sort {path}", ["把 {path} 按行排序", "{path} 内容排序一下"], tags=["text"]),
    Tool("uniq_count", "统计重复行并计数", [Arg("path", "path", "目标文件")],
         "sort {path} | uniq -c | sort -rn", ["统计 {path} 里每行出现次数", "{path} 有哪些重复行"],
         tags=["text"]),
    Tool("word_count", "统计字符/词/行数", [Arg("path", "path", "目标文件")],
         "wc {path}", ["{path} 有多少字", "统计 {path} 的字符数和行数"], tags=["text"]),
    Tool("cut_columns", "按分隔符取指定列",
         [Arg("path", "path", "目标文件"), Arg("sep", "choice", "分隔符", choices=[":", ",", " ", "|", "\t"]),
          Arg("col", "int", "列号", minimum=1, maximum=10)],
         "cut -d'{sep}' -f{col} {path}", ["取 {path} 的第 {col} 列", "把 {path} 按 {sep} 分隔后取第 {col} 列"],
         tags=["text"]),
    Tool("diff_files", "比较两个文件的差异",
         [Arg("a", "path", "文件 A"), Arg("b", "path", "文件 B")],
         "diff {a} {b}", ["比较 {a} 和 {b} 有什么不同", "{a} 和 {b} 的差异在哪"],
         tags=["text"]),

    # ---------------- 代码执行 ----------------
    Tool("python_run", "运行一段 Python 代码并返回输出",
         [Arg("code", "cmd", "要执行的 Python 代码")],
         "python -c {code}", ["用 Python 算一下 {code}", "执行这段 Python：{code}",
                               "写个 Python 脚本做 {code}"],
         level="exec", tags=["py"]),

    # ---------------- 系统信息 ----------------
    Tool("disk_free", "查看磁盘剩余空间", [], "df -h .",
         ["磁盘还剩多少空间", "看看磁盘使用情况", "查看剩余磁盘"], tags=["sys"]),
    Tool("list_process", "查看当前运行的进程", [], "ps -ef | head -n 20",
         ["现在有哪些进程在跑", "看看系统进程"], tags=["sys"]),
    Tool("whoami", "查看当前用户", [], "whoami", ["我是谁", "当前是哪个用户"], tags=["sys"]),
    Tool("which_cmd", "查询命令的安装路径", [Arg("name", "choice", "命令名",
                                                choices=["python", "bash", "grep", "find", "git", "curl"])],
         "which {name}", ["{name} 装在哪", "{name} 命令的路径是什么"], tags=["sys"]),
    Tool("date_now", "查看当前时间", [], "date '+%Y-%m-%d %H:%M:%S'",
         ["现在几点了", "看一下当前时间", "今天几号"], tags=["sys"]),
    Tool("env_get", "读取环境变量", [Arg("name", "choice", "变量名",
                                        choices=["HOME", "PATH", "USER", "PWD"])],
         "printenv {name}", ["环境变量 {name} 是什么", "看看 {name} 的值"], tags=["sys"]),
    Tool("count_files", "统计目录下文件数量", [Arg("path", "path", "目标目录")],
         "find {path} -type f | wc -l", ["{path} 下有多少个文件", "数一下 {path} 里的文件数"], tags=["dir"]),
    Tool("top_files", "列出目录中最大的若干文件",
         [Arg("path", "path", "目标目录"), Arg("n", "int", "数量", minimum=1, maximum=20)],
         "du -a {path} | sort -rn | head -n {n}", ["列出 {path} 里最大的 {n} 个文件",
                                                    "{path} 中占空间最多的前 {n} 个文件"],
         emits_paths=True, tags=["dir"]),
]

TOOL_MAP: dict[str, Tool] = {t.name: t for t in TOOLS}


# ---------------------------------------------------------------------------
# 渲染与解析
# ---------------------------------------------------------------------------
def _sed_escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace("/", "\\/").replace("&", "\\&")


def render_command(tool: Tool, args: dict[str, Any]) -> str:
    """把结构化调用渲染成 POSIX shell 命令。

    注意：choice 参数来自固定白名单（含 " "、":" 这类字符），不能走 shlex；
    sed 的 pattern/replacement 需要单独转义，否则嵌套引号会把命令打断。
    """
    if tool.name == "replace_text":
        return (
            f"sed -i 's/{_sed_escape(args['old'])}/{_sed_escape(args['new'])}/g' "
            f"{shlex.quote(args['path'])}"
        )

    filled: dict[str, str] = {}
    for a in tool.args:
        if a.name in args:
            v = args[a.name]
            if isinstance(v, bool):
                filled[a.name] = "1" if v else "0"
            elif a.type == "choice":
                filled[a.name] = str(v)
            elif a.type == "int":
                filled[a.name] = str(int(v))
            else:
                filled[a.name] = shlex.quote(str(v))
        elif a.required:
            raise ValueError(f"{tool.name} 缺少必填参数 {a.name}")
        else:
            filled[a.name] = "10"  # 可选参数的默认行数

    return tool.template.format(**filled)


def parse_call(text: str) -> tuple[str, dict[str, Any]]:
    """解析 <|tool_call|> 之后的 JSON。容忍模型常见的尾随噪声。"""
    s = text.strip()
    start, end = s.find("{"), s.rfind("}")
    if start < 0 or end < 0:
        raise ValueError("调用体不是合法 JSON 对象")
    obj = json.loads(s[start:end + 1])
    name = obj.get("name") or obj.get("tool")
    args = obj.get("args") or obj.get("arguments") or {}
    if not isinstance(name, str) or name not in TOOL_MAP:
        raise ValueError(f"未知工具: {name!r}")
    if not isinstance(args, dict):
        raise ValueError("args 必须是对象")
    return name, args


def validate_args(tool: Tool, args: dict[str, Any]) -> str | None:
    """返回错误信息，None 表示通过。执行壳与数据合成共用。"""
    known = {a.name for a in tool.args}
    extra = set(args) - known
    if extra:
        return f"存在未知参数: {sorted(extra)}"
    for a in tool.args:
        if a.required and a.name not in args:
            return f"缺少参数 {a.name}"
        if a.name not in args:
            continue
        v = args[a.name]
        if a.type == "int":
            if not isinstance(v, int) or isinstance(v, bool):
                return f"参数 {a.name} 必须是整数"
            if a.minimum is not None and v < a.minimum:
                return f"参数 {a.name} 不能小于 {a.minimum}"
            if a.maximum is not None and v > a.maximum:
                return f"参数 {a.name} 不能大于 {a.maximum}"
        elif a.type in ("path", "text", "pattern", "cmd"):
            if not isinstance(v, str) or not v:
                return f"参数 {a.name} 必须是非空字符串"
            if "\x00" in v:
                return f"参数 {a.name} 含非法字符"
        elif a.type == "choice":
            if v not in (a.choices or []):
                return f"参数 {a.name} 必须是 {a.choices} 之一"
        elif a.type == "bool":
            if not isinstance(v, bool):
                return f"参数 {a.name} 必须是布尔值"
    return None


def tool_manifest() -> str:
    """给提示词/文档用的工具清单（不含模板）。"""
    out = []
    for t in TOOLS:
        sig = ", ".join(
            f"{a.name}:{a.type}" + ("=" + "|".join(a.choices) if a.choices else "")
            + ("" if a.required else "?")
            for a in t.args
        )
        out.append(f"- {t.name}({sig}) — {t.desc}")
    return "\n".join(out)


if __name__ == "__main__":
    print(f"工具数: {len(TOOLS)}")
    print(f"协议 token: {PROTOCOL_TOKENS}")
    print(tool_manifest())
