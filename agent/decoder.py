"""受约束解码：让 40M 模型永远吐不出非法 tool_call。

为什么必须做：
40M 模型自由生成 JSON 时，括号、引号、逗号几乎必错一处（实测单步合法率约 60%）。
把语法检查前移到采样阶段，格式错误直接不可能发生，模型只剩"选对工具和参数"这一件事。

做法：
把一次调用拆成一串"段"（固定字面量 / 工具名 / 参数值），
每步只允许能让当前段继续合法的字符所对应的 token。
byte-level 词表里 256 个单字节 token 始终存在，所以任何字面量都能逐字节落地。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from tools.schema import TOOL_MAP, Tool


# ---------------------------------------------------------------------------
# 段
# ---------------------------------------------------------------------------
@dataclass
class Step:
    kind: str                 # lit | tool | enum | int | free
    text: str = ""
    arg: str | None = None
    options: tuple[str, ...] = ()
    lo: int = 1
    hi: int = 100
    max_len: int = 80
    allow_structural: bool = False

    def allowed(self, buf: str) -> set[str]:
        if self.kind == "lit":
            if not self.text.startswith(buf) or len(buf) >= len(self.text):
                return set()
            return {self.text[len(buf)]}
        if self.kind in ("tool", "enum"):
            chars: set[str] = set()
            # 已经拼成完整候选时，只允许用引号收尾
            if buf in self.options:
                chars.add('"')
            chars |= {o[len(buf)] for o in self.options
                      if o.startswith(buf) and len(buf) < len(o)}
            return chars
        if self.kind == "int":
            # 只放开还能落在 [lo, hi] 内的数字，否则模型会无限吐数字
            return {d for d in "0123456789"
                    if (buf + d).isdigit() and int(buf + d) <= self.hi}
        # 自由文本：禁止破坏 JSON 的字符。
        chars = {chr(i) for i in range(32, 127)} | {chr(i) for i in range(0x4E00, 0x9FA6)}
        chars -= {'"', "\\", "\n", "\r", "\t"}
        if self.allow_structural:
            # 代码参数（python_run 的 code）里允许写 dict/list，但右括号必须配对：
            # 否则模型会在字符串里直接吐 "}}" 想收尾 JSON，结果把调用截断。
            chars -= {"{", "}", "[", "]"}
            chars |= {"{", "[", ","}
            if buf.count("{") > buf.count("}"):
                chars.add("}")
            if buf.count("[") > buf.count("]"):
                chars.add("]")
        else:
            chars -= {"{", "}", "[", "]", ","}
        if len(buf) == 0:
            return chars
        return chars | {'"'}

    def done(self, buf: str) -> bool:
        """不含收尾引号的状态判断（引号由外层统一吃）。"""
        if self.kind == "lit":
            return buf == self.text
        if self.kind in ("tool", "enum"):
            return buf in self.options
        if self.kind == "int":
            return buf.isdigit() and self.lo <= int(buf) <= self.hi
        return 0 < len(buf) <= self.max_len


# 这些工具的目标路径允许不存在（新建），其余工具的路径必须来自环境清单
FREE_PATH_TOOLS = {"mkdir", "touch", "write_file", "append_file"}

# 路径参数按语义限定候选类型：目录类工具不该选到文件。
# 不区分的话，模型会把 "docs 占多大空间" 落到 docs/changelog.md 上。
DIR_TOOLS = {"list_dir", "tree", "du_dir", "find_name", "find_ext", "find_large",
             "find_empty", "delete_dir", "count_files", "top_files"}
FILE_TOOLS = {"stat_file", "file_type", "count_lines", "read_head", "read_tail",
              "read_range", "cat_file", "grep_text", "grep_count", "delete_file",
              "truncate_file", "sort_file", "uniq_count", "word_count",
              "cut_columns", "diff_files", "replace_text"}


class LockedCall:
    """维护 {"name":"X","args":{...}} 的生成进度，并给出当前允许的字符集。

    path_candidates：执行壳提供的真实文件清单。给了之后，路径参数变成
    "从现有文件里选一个"，而不是让 37M 模型凭记忆拼路径——这是小模型能
    准确落地的关键，路径本来就该来自环境而不是权重。
    """

    def __init__(self, tool_hint: Tool | None = None, path_candidates: list[str] | None = None,
                 path_root: str | None = None):
        self.plan: list[Step] = [Step("lit", '{"name":"'), Step("tool", options=tuple(TOOL_MAP.keys())),
                                 Step("lit", ',"args":{')]
        self.buf = ""
        self.values: dict = {}
        self.tool: Tool | None = tool_hint
        self.finished = False
        self.text = ""
        self.path_candidates = path_candidates or []
        self.path_root = path_root

    def _paths_for(self, tool_name: str) -> list[str]:
        """按工具语义把候选路径筛成目录或文件；不存在的路径（用户点名的）保留。"""
        if not self.path_root:
            return self.path_candidates
        kind = "dir" if tool_name in DIR_TOOLS else ("file" if tool_name in FILE_TOOLS else None)
        if kind is None:
            return self.path_candidates
        out = []
        for c in self.path_candidates:
            full = Path(self.path_root) / c
            if not full.exists():
                out.append(c)          # 用户点名的路径，允许它不存在
            elif kind == "dir" and full.is_dir():
                out.append(c)
            elif kind == "file" and full.is_file():
                out.append(c)
        return out or self.path_candidates

    # ---- 当前允许字符 ----
    def allowed(self) -> set[str]:
        if self.finished:
            return set()
        step = self.plan[0]
        chars = step.allowed(self.buf)
        # 整数没有引号收尾，必须把"下一个字面量的首字符"（} 或 ,）也算作合法终止，
        # 否则模型只能无限吐数字，最后被强制 <|end|> 截断——参数就丢了。
        if step.kind == "int" and self.buf.isdigit() and step.lo <= int(self.buf) <= step.hi:
            if len(self.plan) > 1 and self.plan[1].kind == "lit" and self.plan[1].text:
                chars = chars | {self.plan[1].text[0]}
        return chars

    # ---- 吃字符 ----
    def feed(self, piece: str) -> None:
        """逐字符推进：一个 token 可能吐出 '":"' 甚至跨段片段，
        整段追加会让缓冲区越过段边界，状态机随即错位。"""
        for ch in piece:
            self._feed_char(ch)
            if self.finished:
                return

    def _feed_char(self, ch: str) -> None:
        if self.finished:
            return
        step = self.plan[0]

        # 整数没有结束引号，遇到第一个非数字就收尾，并把这个字符交给下一段重新处理
        if step.kind == "int":
            if ch in "0123456789":
                self.text += ch
                self.buf += ch
                return
            if not (self.buf.isdigit() and step.lo <= int(self.buf) <= step.hi):
                raise ValueError(f"整数非法: {self.raw_call()}")
            self.values[step.arg] = int(self.buf)
            self.buf = ""
            self.plan.pop(0)
            self._feed_char(ch)
            return

        self.text += ch

        if step.kind == "lit":
            self.buf += ch
            if not step.text.startswith(self.buf):
                raise ValueError(f"字面量失配: {self.raw_call()}")
            if step.text == self.buf:
                self.plan.pop(0)
                self.buf = ""
            return

        if ch == '"':
            if not step.done(self.buf):
                raise ValueError(f"非法收尾: {self.raw_call()}")
            if step.kind == "tool":
                self.tool = TOOL_MAP[self.buf]
                self.plan.pop(0)          # 工具名槽位
                self._expand_args()       # 参数插到 '","args":{' 之后
            else:
                self.values[step.arg] = int(self.buf) if step.kind == "int" else self.buf
                self.plan.pop(0)
            self.buf = ""
            return

        self.buf += ch
        if len(self.buf) > step.max_len + 8:
            raise ValueError("参数过长")

    def _expand_args(self) -> None:
        assert self.tool is not None
        req = [a for a in self.tool.args if a.required]
        new: list[Step] = []
        for i, a in enumerate(req):
            if i:
                new.append(Step("lit", ","))
            # 开引号必须算进字面量，否则 JSON 缺引号；int 是裸数字，不加引号
            if a.type == "int":
                new.append(Step("lit", f'"{a.name}":'))
                new.append(Step("int", arg=a.name, lo=a.minimum or 1, hi=a.maximum or 100))
            else:
                new.append(Step("lit", f'"{a.name}":"'))
                if a.type == "choice":
                    new.append(Step("enum", arg=a.name, options=tuple(a.choices or ())))
                elif (a.type == "path" and self.path_candidates
                      and self.tool.name not in FREE_PATH_TOOLS and a.name != "dst"):
                    new.append(Step("enum", arg=a.name,
                                    options=tuple(self._paths_for(self.tool.name))))
                else:
                    is_code = a.type == "cmd"
                    new.append(Step("free", arg=a.name, max_len=200 if is_code else 80,
                                    allow_structural=is_code))
        new.append(Step("lit", "}}"))
        # plan[0] 是 '","args":{'，参数要排在它后面
        self.plan[1:1] = new

    def advance(self) -> bool:
        """所有段都走完即为结束。"""
        if not self.plan:
            self.finished = True
        return self.finished

    def raw_call(self) -> str:
        return self.text


# ---------------------------------------------------------------------------
# 解码器
# ---------------------------------------------------------------------------
class ConstrainedDecoder:
    """logits 过滤器，配合 MiniGPT.generate 使用。

    free 段：不约束，让模型决定是"说话"还是"发调用"；
    一旦采样到 <|tool_call|>，切到 json 段做逐字符锁定；
    调用闭合后强制输出 <|end|>。
    """

    def __init__(self, tok, call_token: str = "<|tool_call|>", end_token: str = "<|end|>",
                 path_candidates: list[str] | None = None, path_root: str | None = None):
        self.tok = tok
        self.path_candidates = path_candidates or []
        self.path_root = path_root
        self.call_id = tok.convert_tokens_to_ids(call_token)
        self.end_id = tok.convert_tokens_to_ids(end_token)
        self.texts = [tok.decode([i]) for i in range(tok.vocab_size)]
        # 特殊 token 的文本（如 "<|end|>"）由可打印字符组成，会伪装成合法自由文本；
        # 不排除掉，模型就能在参数字符串中间直接吐 <|end|>，把调用截断成半个 JSON。
        # 注意不能只信 all_special_ids：tokenizer_v3 里协议 token 没登记为 special，
        # 只有 0~4 在里面，<|end|> 漏网——所以按文本形状再兜一层。
        self.special_ids = set(getattr(tok, "all_special_ids", []) or [])
        self.quote_ids = [i for i, t in enumerate(self.texts) if t == '"']
        self._opt_cache: dict = {}
        self.buf = 0            # free 段已生成的字符数（用于判定是否出现调用标记）
        self.mode = "free"
        self.lock: LockedCall | None = None
        self.saw_call = False
        self.error: str | None = None

    # ---- 每个采样步调用 ----
    def __call__(self, step: int, logits: torch.Tensor) -> torch.Tensor:
        if self.mode == "free":
            return logits
        if self.mode == "closed":
            return self._only(logits, self.end_id)
        if self.error:
            return self._only(logits, self.end_id)
        try:
            step = self.lock.plan[0]
            chars = self.lock.allowed() if self.lock else set()
        except Exception as e:  # noqa: BLE001
            self.error = str(e)
            return self._only(logits, self.end_id)
        if not chars:
            return self._only(logits, self.end_id)

        # 工具名/枚举值按 token 约束，而不是按字符：
        # 训练时 "append_file" 是少数几个 token，逐字符喂进去属于分布外输入，
        # 模型会退化成默认值。这里只要求 "buf + token" 仍是某个候选的前缀。
        if step.kind == "lit":
            return self._only_ids(logits, self._lit_ids(step.text[len(self.lock.buf):]))

        if step.kind in ("tool", "enum"):
            ids = self._option_ids(step.options, self.lock.buf)
            if self.lock.buf in step.options:
                ids = ids + self.quote_ids
            return self._only_ids(logits, ids)

        if step.kind == "int":
            return self._only_ids(logits, self._int_ids(step, self.lock.buf))

        return self._only_ids(logits, self._ids_for(chars, multi=(step.kind == "free")))

    # ---- 每个 token 采样后调用 ----
    def observe(self, token_id: int) -> None:
        if self.mode == "free":
            if token_id == self.call_id:
                self.mode = "json"
                self.lock = LockedCall(path_candidates=self.path_candidates,
                                       path_root=self.path_root)
                self.saw_call = True
            return
        if self.mode == "json":
            if token_id == self.end_id:
                self.mode = "closed"
                return
            piece = self.texts[token_id]
            try:
                self.lock.feed(piece)
            except ValueError as e:
                self.error = str(e)
                self.mode = "closed"
                return
            if self.lock.advance():
                self.mode = "closed"

    # ---- 结果 ----
    @property
    def call(self) -> tuple[str, dict] | None:
        """只有状态机完整走完才算一次有效调用，半截不算。"""
        if self.lock and self.lock.tool and self.lock.finished:
            return self.lock.tool.name, self.lock.values
        return None

    # ---- 内部 ----
    def _lit_ids(self, remain: str) -> list[int]:
        """字面量也可以整段吐出来：训练时 '{"name":"' 可能就是 1~2 个 token。"""
        hit = self._opt_cache.get(("lit", remain))
        if hit is not None:
            return hit
        ids = [i for i, t in enumerate(self.texts) if t and remain.startswith(t)]
        self._opt_cache[("lit", remain)] = ids
        return ids

    def _int_ids(self, step: Step, buf: str) -> list[int]:
        """整数按 token 粒度放开（训练时 "20" 就是一个 token，逐字符喂属于分布外），
        同时保留单字符的 , 或 } 作为收尾。"""
        term = ""
        if len(self.lock.plan) > 1 and self.lock.plan[1].kind == "lit":
            term = self.lock.plan[1].text[:1]
        ok_close = bool(buf) and buf.isdigit() and step.lo <= int(buf) <= step.hi
        out = []
        for i, t in enumerate(self.texts):
            if not t or i in self.special_ids:
                continue
            if t.isdecimal() and t.isascii():
                v = buf + t
                if v.isdecimal() and int(v) <= step.hi:
                    out.append(i)
            elif ok_close and term and t == term:
                out.append(i)
        return out

    def _option_ids(self, options: tuple[str, ...], buf: str) -> list[int]:
        """允许 "buf + token 文本" 仍是候选前缀的所有 token。按 (候选项, buf) 缓存。"""
        key = (options, buf)
        hit = self._opt_cache.get(key)
        if hit is not None:
            return hit
        ids = []
        for i, t in enumerate(self.texts):
            if not t:
                continue
            p = buf + t
            if any(o.startswith(p) for o in options):
                ids.append(i)
        self._opt_cache[key] = ids
        return ids

    def _ids_for(self, chars: set[str], multi: bool = False) -> list[int]:
        """单字符 token 永远可用（byte-level 保证 256 个都在）；
        只有自由文本段才放开多字符 token 提速，且不允许 token 里夹引号。"""
        out = []
        for i, t in enumerate(self.texts):
            if not t or i in self.special_ids:
                continue
            if t.startswith("<|") and t.endswith("|>"):
                continue  # 协议 token 一律不许出现在参数值里
            if len(t) == 1:
                if t in chars:
                    out.append(i)
            elif multi and '"' not in t and all(c in chars for c in t):
                out.append(i)
        return out

    def _only(self, logits: torch.Tensor, tid: int) -> torch.Tensor:
        return self._only_ids(logits, [tid])

    def _only_ids(self, logits: torch.Tensor, ids: list[int]) -> torch.Tensor:
        if not ids:
            return logits
        idx = torch.tensor(ids, device=logits.device, dtype=torch.long)
        neg = torch.full_like(logits, float("-inf"))
        neg.scatter_(-1, idx.unsqueeze(0).expand(logits.shape[0], -1), 0.0)
        return logits + neg


if __name__ == "__main__":
    # 直接喂一段标准 JSON，检查状态机能否正确解析出参数值
    import json

    cases = [
        ("list_dir", {"path": "src"}),
        ("find_large", {"path": "src", "min_mb": 3}),
        ("cut_columns", {"path": "data/users.csv", "sep": ",", "col": 2}),
        ("python_run", {"code": "print(1)"}),
        ("disk_free", {}),
        ("read_head", {"path": "README.md", "lines": 5}),
    ]
    for name, args in cases:
        tool = TOOL_MAP[name]
        full = {"name": name, "args": {a.name: args[a.name] for a in tool.args if a.required}}
        payload = json.dumps(full, ensure_ascii=False, separators=(",", ":"))
        lk = LockedCall()
        for ch in payload:
            lk.feed(ch)
        lk.advance()
        ok = lk.tool and lk.tool.name == name and lk.values == full["args"] and lk.finished
        print(f"{'OK ' if ok else 'BAD'} {name:12s} -> {lk.tool.name if lk.tool else None} {lk.values} finished={lk.finished}")
