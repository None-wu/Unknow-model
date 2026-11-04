"""增量 SFT 数据：补三类缺口，合并进 v4。

1. python_run 的自然语言 → 代码：原数据指令里直接写代码（"执行这段 Python：print(2**10)"），
   模型只学会复制代码；换成"用 python 算一下 2 的 10 次方"这类说法，逼它自己写代码。
   输出一律真跑一遍 python -c 拿到，不编。
2. count_lines 的口语说法（"一共几行"曾误判成 cut_columns）。
3. 长标识符参数保真（ACCEPTANCE_MARK 被写成 AcCEPTANCE_MARK）。
4. 身份对话（避免每轮 SFT 后再补丁）。

用法：
  ./.venv/Scripts/python.exe tools/extra_sft.py --src data/tool_sft.jsonl --out data/tool_sft_v4.jsonl
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# (自然语言描述, python 表达式) —— 描述里不出现代码
PY_PAIRS: list[tuple[str, str]] = [
    ("2 的 10 次方", "print(2**10)"),
    ("2 的 16 次方", "print(2**16)"),
    ("3 的 4 次方", "print(3**4)"),
    ("5 的平方", "print(5**2)"),
    ("7 的立方", "print(7**3)"),
    ("1 到 100 的和", "print(sum(range(1, 101)))"),
    ("1 到 10 的和", "print(sum(range(1, 11)))"),
    ("10 的阶乘", "print(1*2*3*4*5*6*7*8*9*10)"),
    ("17 除以 5 的余数", "print(17 % 5)"),
    ("100 除以 7 的整数部分", "print(100 // 7)"),
    ("15 加 27 是多少", "print(15 + 27)"),
    ("48 减 19 是多少", "print(48 - 19)"),
    ("12 乘以 8 是多少", "print(12 * 8)"),
    ("81 除以 9 是多少", "print(81 / 9)"),
    ("3.14159 保留两位小数", "print(round(3.14159, 2))"),
    ("2.71828 保留一位小数", "print(round(2.71828, 1))"),
    ("9 的平方根", "print(9**0.5)"),
    ("16 的平方根", "print(16**0.5)"),
    ("字符串 hello 的长度", "print(len('hello'))"),
    ("字符串 unknow 的长度", "print(len('unknow'))"),
    ("列表 3 1 2 排序后的结果", "print(sorted([3, 1, 2]))"),
    ("列表 5 9 1 4 里的最大值", "print(max([5, 9, 1, 4]))"),
    ("列表 5 9 1 4 里的最小值", "print(min([5, 9, 1, 4]))"),
    ("1 到 20 里的偶数", "print([x for x in range(1, 21) if x % 2 == 0])"),
    ("1 到 10 里的奇数", "print([x for x in range(1, 11) if x % 2])"),
    ("把 unknow 全部转成大写", "print('unknow'.upper())"),
    ("把 HELLO 全部转成小写", "print('HELLO'.lower())"),
    ("反转字符串 abcdef", "print('abcdef'[::-1])"),
    ("统计 hello world 里有几个 o", "print('hello world'.count('o'))"),
    ("'a,b,c' 按逗号切开", "print('a,b,c'.split(','))"),
    ("把 2026 年换算成天数（按 365 天）", "print(2026*365)"),
    ("圆的半径 3 算面积（π 取 3.14）", "print(3.14*3*3)"),
    ("长方形的长 8 宽 5 算面积", "print(8*5)"),
    ("摄氏 25 度换算华氏度", "print(25*9/5+32)"),
    ("随机数固定种子 42 生成一个整数", "print(__import__('random').Random(42).randint(0, 100))"),
    ("当前 python 版本号", "print(__import__('sys').version.split()[0])"),
]

PY_PHRASES = [
    "用 python 算一下 {d}",
    "python 里算算 {d}",
    "帮我用 python 求 {d}",
    "跑段 python 看看 {d}",
    "用 python 计算：{d}",
    "写个 python 输出 {d}",
]

LINE_PHRASES = [
    "{p} 一共几行",
    "{p} 有多少行",
    "{p} 的行数是多少",
    "{p} 总共多少行啊",
    "数一下 {p} 有几行",
]

LONG_TEXTS = [
    "ACCEPTANCE_MARK",
    "BUILD_ID_2026",
    "TODO_QUEUE_V2",
    "DeployKey_9f3A",
    "RELEASE_CANDIDATE_1",
    "API_TOKEN_PLACEHOLDER",
    "v2.4.1-hotfix",
    "EnableCache=true",
]

IDENTITY = [
    ("你是谁", "我是 Unknow，一个 62M 参数的小型命令行助手模型。我可以帮你查看目录、搜索文本、读写文件和执行简单命令。"),
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


def py_out(code: str) -> str:
    try:
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=10)
        return (r.stdout or "").strip() or "(无输出)"
    except Exception:  # noqa: BLE001
        return "(无输出)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/tool_sft.jsonl",
                    help="run.py synth 的产出（基础轨迹）")
    ap.add_argument("--out", default="data/tool_sft_final.jsonl",
                    help="合并后的最终语料，训练默认读这个")
    a = ap.parse_args()

    base = [line for line in (ROOT / a.src).open(encoding="utf-8")]
    extra: list[dict] = []

    # 1) python_run 自然语言 -> 代码（真执行取输出）
    for desc, code in PY_PAIRS:
        out = py_out(code)
        for phr in PY_PHRASES[:3]:
            u = phr.format(d=desc)
            extra.append({"kind": "single", "events": [
                {"t": "user", "text": u},
                {"t": "call", "name": "python_run", "args": {"code": code},
                 "output": out, "ok": True},
                {"t": "assistant", "text": f"结果如下：{out}"},
            ], "meta": {"cmd": [f"python -c '{code}'"]}})

    # 2) count_lines 口语说法
    for path in ("logs/error.log", "build/app.log", "docs/guide.md", "data/users.csv"):
        for phr in LINE_PHRASES:
            u = phr.format(p=path)
            extra.append({"kind": "single", "events": [
                {"t": "user", "text": u},
                {"t": "call", "name": "count_lines", "args": {"path": path},
                 "output": "24", "ok": True},
                {"t": "assistant", "text": "共 24 项。"},
            ], "meta": {"cmd": []}})

    # 3) 长标识符保真
    for text in LONG_TEXTS:
        for path, verb in (("build/notes.txt", "结尾加上"), ("docs/guide.md", "末尾追加")):
            extra.append({"kind": "single", "events": [
                {"t": "user", "text": f"给 {path} {verb} {text}"},
                {"t": "call", "name": "append_file", "args": {"path": path, "text": text},
                 "output": "", "ok": True},
                {"t": "assistant", "text": f"已写入 {path}。"},
            ], "meta": {"cmd": []}})

    # 4) 身份对话（×8：混在 2 万条里单份权重太低，学不牢）
    for _ in range(8):
        for q, ans in IDENTITY:
            extra.append({"kind": "chat", "events": [
                {"t": "user", "text": q}, {"t": "assistant", "text": ans}]})

    # 5) cut_columns：中文分隔符说法 -> 符号（原数据只在指令里写符号本身）
    SEP_WORDS = [("逗号", ","), ("制表符", "\t"), ("Tab", "\t"),
                 ("空格", " "), ("冒号", ":"), ("竖线", "|")]
    for word, sep in SEP_WORDS:
        for path, col in (("data/users.csv", 2), ("build/app.log", 1), ("docs/guide.md", 3)):
            for phr in ("把 {p} 按 {w} 切开取第 {c} 列", "{p} 用 {w} 分隔，要第 {c} 列",
                        "按{w}分割 {p} 拿第{c}列"):
                u = phr.format(p=path, w=word, c=col)
                extra.append({"kind": "single", "events": [
                    {"t": "user", "text": u},
                    {"t": "call", "name": "cut_columns",
                     "args": {"path": path, "sep": sep, "col": col},
                     "output": "", "ok": True},
                    {"t": "assistant", "text": f"结果如下：（第 {col} 列内容）"},
                ], "meta": {"cmd": []}})

    # 6) 真实项目结构 -> 摘要：治"陌生目录复读 / 幻觉再调工具"
    #    训练集里的目录都是 8 项左右的小夹具，真实项目动辄几十项、名字陌生，
    #    模型就退化成抄路径、复读、甚至乱 mkdir。这里补不同规模/命名的真实感目录。
    PROJECTS: list[tuple[list[str], dict[str, list[str]]]] = [
        (["LICENSE", "README.md", "package.json", "vite.config.ts", "src", "public", "dist", "node_modules"],
         {"src": ["App.tsx", "main.tsx", "components", "pages", "hooks", "utils"],
          "public": ["favicon.ico", "logo.svg"]}),
        (["README.md", "requirements.txt", "setup.py", "train.py", "configs", "data", "checkpoints", "logs"],
         {"configs": ["base.yaml", "small.yaml"], "data": ["train.txt", "val.txt"]}),
        (["LICENSE", "README.md", "CMakeLists.txt", "include", "src", "tests", "build", "third_party"],
         {"src": ["main.cpp", "engine.cpp", "util.cpp"], "include": ["engine.h", "util.h"]}),
        (["README.md", "pom.xml", "src", "target", ".gitignore", "docs"],
         {"src": ["main", "test"], "docs": ["design.md", "api.md"]}),
        (["README.md", "package.json", "tsconfig.json", "src", "test", "dist", ".github"],
         {"src": ["index.ts", "router.ts", "service.ts"], "test": ["unit.test.ts"]}),
        (["README.md", "go.mod", "go.sum", "cmd", "internal", "pkg", "api", "Makefile"],
         {"cmd": ["server"], "internal": ["config", "store"], "api": ["v1"]}),
        (["README.md", "main.py", "app", "static", "templates", "tests", "migrations", ".env.example"],
         {"app": ["models.py", "views.py", "forms.py"], "templates": ["index.html", "base.html"]}),
    ]
    STRUCT_PHRASES = ["看看这个项目的结构", "这个项目里都有啥", "帮我看看项目结构，不要修改内容",
                      "梳理一下这个项目的目录结构", "这个仓库的结构是怎样的"]

    def make_listing(top: list[str], sub: dict[str, list[str]]) -> str:
        lines = ["."]
        for t in top:
            lines.append(f"./{t}")
            for c in sub.get(t, []):
                lines.append(f"./{t}/{c}")
        return "\n".join(sorted(lines))

    for top, sub in PROJECTS:
        listing = make_listing(top, sub)
        names = [t for t in top if "." in t][:4] or top[:4]
        n_sub = sum(len(v) for v in sub.values())
        summary = (f"项目根目录有 {'、'.join(names)} 等 {len(top)} 项"
                   + (f"，{len(sub)} 个子目录下共 {n_sub} 个文件" if sub else "")
                   + "。")
        for phr in STRUCT_PHRASES:
            extra.append({"kind": "single", "events": [
                {"t": "user", "text": phr},
                {"t": "call", "name": "tree", "args": {"path": "."},
                 "output": listing, "ok": True},
                {"t": "assistant", "text": summary},
            ], "meta": {"cmd": ["find . -maxdepth 2 | sort"]}})

    # 7) list_dir 的"里面有些什么文件"类说法（曾被误判成 find_ext）
    for d in ("build", "src", "docs", "data", "logs", "."):
        for phr in ("{d} 里面有些什么文件", "{d} 里面都有什么", "{d} 下面有些啥",
                    "看看 {d} 里有什么文件"):
            u = phr.format(d=d)
            extra.append({"kind": "single", "events": [
                {"t": "user", "text": u},
                {"t": "call", "name": "list_dir", "args": {"path": d},
                 "output": "README.md\napp.log\nnotes.txt", "ok": True},
                {"t": "assistant", "text": "目录里有 README.md、app.log、notes.txt，共 3 项。"},
            ], "meta": {"cmd": []}})

    # 8) 知识问答 -> 拒答：模型只学过文件操作，通用知识问题（"Apache-2.0 是什么"）
    #    在训练分布外，会退化成乱码/幻觉调用。教它礼貌承认不会，把话题拉回本行。
    KNOW_TOPICS = [
        "Apache-2.0 授权", "CC-BY-NC 授权", "MIT 协议", "GPL 协议",
        "HTTP 协议", "TCP 和 UDP", "机器学习", "深度学习", "区块链",
        "量子计算", "相对论", "进化论", "光合作用", "黑洞",
        "Python 的 GIL", "Docker 的原理", "Kubernetes", "TCP 三次握手",
        "敏捷开发", "微服务架构", "操作系统内核", "编译原理",
        "美联储加息", "通货膨胀", "股票市盈率", "汇率",
        "唐诗和宋词", "文艺复兴", "第二次世界大战", "长城的历史",
    ]
    KNOW_PATTERNS = [
        "{t}是什么", "什么是{t}", "{t}是什么意思", "解释一下{t}",
        "给我讲讲{t}", "{t}是怎么回事",
    ]
    COMPARE = [("Apache-2.0", "MIT"), ("TCP", "UDP"), ("HTTP", "HTTPS"),
               ("进程", "线程"), ("Python", "Java"), ("深度学习", "机器学习")]
    WHY = ["为什么天是蓝的", "为什么会下雨", "为什么猫会叫", "为什么代码会有 bug 这么难学",
           "为什么地球是圆的"]
    CHAT_OOD = [("讲个笑话", "我讲不好笑话，只会操作文件和命令。要不要我帮你看点项目里的东西？"),
                ("写一首诗", "写诗不是我的强项，我是文件操作助手。有文件要改、要查的尽管说。"),
                ("翻译一段英文", "翻译不在我的能力范围。如果你要把翻译结果写进某个文件，可以先把译文给我，我帮你写入。"),
                ("今天股市怎么样", "我查不了行情，只能操作本地文件和跑命令。")]
    REFUSE_KNOW = [
        "这个知识问答超出了我的能力，我只会文件/目录操作和运行命令。",
        "我是文件操作助手，通用知识不在我的能力范围内。如果你想查的是项目里的内容，我可以帮你 grep 一下。",
        "这个我答不了——我是个小型工具模型，只会查目录、搜文本、读写文件和跑命令。",
        "知识问答不是我的活儿。有文件要处理的话，直接说文件名和想做的事。",
        "这个问题我回答不了。我的能力是：列目录、看文件、搜索内容、简单计算和文件修改。",
    ]

    def _know(kind_q: str, ans: str) -> dict:
        return {"kind": "refuse", "events": [
            {"t": "user", "text": kind_q}, {"t": "assistant", "text": ans}]}

    n_know = 0
    for t in KNOW_TOPICS:
        for phr in KNOW_PATTERNS:
            extra.append(_know(phr.format(t=t), REFUSE_KNOW[n_know % len(REFUSE_KNOW)]))
            n_know += 1
    for a_, b_ in COMPARE:
        for phr in ("{a} 和 {b} 有什么区别", "{a}跟{b}的区别是什么"):
            extra.append(_know(phr.format(a=a_, b=b_),
                               REFUSE_KNOW[n_know % len(REFUSE_KNOW)]))
            n_know += 1
    for q in WHY:
        extra.append(_know(q, REFUSE_KNOW[n_know % len(REFUSE_KNOW)]))
        n_know += 1
    for q, ans in CHAT_OOD:
        extra.append(_know(q, ans))
        n_know += 1

    # 7.5) "结构" -> tree 的对照样本：第 7 类的 list_dir 数据会把
    #      "看看 X 这个目录的结构" 带偏成 list_dir（只回文件名、不提目录名）。
    #      明确划界：说"结构" -> tree 递归列出；说"里面有什么" -> list_dir。
    for d, listing in (("src", "./config.json\n./main.py\n./utils.py"),
                       ("build", "./app.log\n./notes.txt\n./output.bin"),
                       ("docs", "./guide.md\n./api.md\n./img"),
                       ("tools", "./schema.py\n./executor.py\n./synth.py"),
                       (".", "./README.md\n./src\n./build\n./docs\n./logs")):
        names = [ln.split("/")[-1] for ln in listing.splitlines()]
        summary = f"{d} 的结构：{'、'.join(names[:5])}，共 {len(names)} 项。"
        for phr in ("看看 {d} 这个目录的结构", "{d} 的目录结构是怎样的",
                    "梳理一下 {d} 的结构", "看下 {d} 的结构，不要修改"):
            u = phr.format(d=d)
            extra.append({"kind": "single", "events": [
                {"t": "user", "text": u},
                {"t": "call", "name": "tree", "args": {"path": d},
                 "output": listing, "ok": True},
                {"t": "assistant", "text": summary},
            ], "meta": {"cmd": [f"find {d} -maxdepth 2 | sort"]}})

    out_path = ROOT / a.out
    with out_path.open("w", encoding="utf-8") as f:
        f.writelines(base)
        for r in extra:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{a.src} {len(base)} 条 + 新增 {len(extra)} 条 -> {a.out}")


if __name__ == "__main__":
    main()
