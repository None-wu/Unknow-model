"""Agent 循环：模型提议 → 约束解码 → 校验 → 执行 → 回灌。

这里不做花活，就三件事：
1. 每一轮只让模型输出"要么说话、要么发一个调用"；
2. 调用体由 ConstrainedDecoder 保证 JSON 合法，参数通过 schema 校验；
3. 真正的可靠性来自执行壳——危险操作要确认，失败把 stderr 原样喂回去，
   模型下一轮看到的是真实报错，而不是自己编的。
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import torch

from agent.decoder import ConstrainedDecoder
from model.model import MiniGPT
from tools.executor import BashWorker, candidates_for, guard_command
from tools.schema import (A_TOKEN, CALL_TOKEN, END_TOKEN, RESULT_TOKEN, T_TOKEN,
                          U_TOKEN, TOOL_MAP, parse_call, render_command, validate_args)

ROOT = Path(__file__).resolve().parent.parent

# 短语级复读截断：repeat_stop 只防单 token 连续重复，防不住
# "目录里有 Eump,目录里有 Eump,..." 这类多 token 短语循环。
_LOOP_RE = re.compile(r"(.{3,48}?)\1{2,}")
_ABS_IN_TEXT = re.compile(r"[A-Za-z]:[\\/][A-Za-z0-9_\-\\/ .()]*")
# 真实项目的 find/ls 输出里 .git、node_modules 这类噪声占一大半，
# 小模型会把它们当正文抄进摘要并陷入循环；喂给模型前先滤掉。
_NOISE_RE = re.compile(r"(/\.git/|/node_modules/|/__pycache__/|/dist/|/target/)")


def filter_noise(text: str) -> str:
    """丢掉版本库/依赖目录的行，最多保留 30 行。"""
    lines = [ln for ln in text.splitlines() if not _NOISE_RE.search(ln)]
    if len(lines) > 30:
        lines = lines[:30] + ["…（其余略）"]
    return "\n".join(lines)


def cut_phrase_loop(text: str) -> tuple[str, bool]:
    """同一个 3~48 字短语连续出现 3 次以上时，只保留第一遍。

    返回 (清理后的文本, 是否发生了复读)。
    """
    m = _LOOP_RE.search(text)
    if m:
        return text[:m.start()] + m.group(1), True
    return text, False


class Agent:
    def __init__(self, ckpt: str = "checkpoints/sft.pt", tokenizer: str = "tokenizer/tokenizer_v3",
                 cwd: str = ".", max_iters: int = 6, timeout: float = 8.0,
                 temperature: float = 0.3, max_new_tokens: int = 260,
                 confirm=None, verbose: bool = True, device: str | None = None,
                 constrained: bool = False):
        from transformers import PreTrainedTokenizerFast
        from trainer.utils import load_ckpt

        self.tok = PreTrainedTokenizerFast.from_pretrained(str(ROOT / tokenizer))
        self.model, ck, self.cfg = load_ckpt(ROOT / ckpt)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()
        self.cwd = Path(cwd).resolve()
        self.max_iters = max_iters
        self.timeout = timeout
        self.temperature = temperature
        self.max_new = max_new_tokens
        self.confirm = confirm
        self.verbose = verbose
        self.constrained = constrained

        self.u_id = self.tok.convert_tokens_to_ids(U_TOKEN)
        self.a_id = self.tok.convert_tokens_to_ids(A_TOKEN)
        self.end_id = self.tok.convert_tokens_to_ids(END_TOKEN)
        self.max_ctx = self.cfg.max_seq_len
        self.ids: list[int] = []
        self.worker = BashWorker().start()

    def close(self) -> None:
        self.worker.close()

    # ---- 编码：与训练时一致，逐段编码后拼接，避免跨边界 merge ----
    def _enc(self, text: str) -> list[int]:
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def reset(self) -> None:
        self.ids = []

    def _truncate(self) -> None:
        if len(self.ids) > self.max_ctx - self.max_new:
            keep = self.max_ctx - self.max_new
            self.ids = self.ids[-keep:]

    # ---- 执行 ----
    def _execute(self, name: str, args: dict) -> tuple[bool, str]:
        tool = TOOL_MAP[name]
        err = validate_args(tool, args)
        if err:
            return False, f"参数不合法：{err}"
        try:
            cmd = render_command(tool, args)
        except Exception as e:  # noqa: BLE001
            return False, f"命令渲染失败：{e}"
        blocked = guard_command(cmd)
        if blocked:
            return False, f"已拦截：{blocked}"
        if tool.level != "read" and self.confirm is not None:
            if not self.confirm(cmd):
                return False, "用户拒绝了这次操作"
        if self.verbose:
            print(f"  [执行] {cmd}")
        res = self.worker.run(cmd, self.cwd, timeout=self.timeout)
        out = res.output or ("（无输出）" if res.ok else "（失败且无输出）")
        return res.ok, out

    # ---- 主循环 ----
    def _generate(self, prompt_ids: list[int], constrained: bool,
                  hint: str = "") -> tuple[str, list[int], tuple | None]:
        """生成一轮。constrained=False 时靠解析+校验兜底，实测准确率更高；
        解析失败再退回语法约束重生成一次。"""
        x = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        if constrained:
            dec = ConstrainedDecoder(self.tok,
                                     path_candidates=candidates_for(self.cwd, hint),
                                     path_root=str(self.cwd))
            out = self.model.generate(
                x, max_new_tokens=self.max_new, temperature=self.temperature,
                top_k=40, stop_ids={self.end_id}, logits_filter=dec, on_token=dec.observe,
            )
            ids = out[0].tolist()
            return self.tok.decode(ids, skip_special_tokens=False), ids, dec.call
        out = self.model.generate(
            x, max_new_tokens=self.max_new, temperature=self.temperature,
            top_k=40, stop_ids={self.end_id},
        )
        ids = out[0].tolist()
        text = self.tok.decode(ids, skip_special_tokens=False)
        call = None
        if CALL_TOKEN in text:
            try:
                call = parse_call(text.split(CALL_TOKEN, 1)[1])
            except Exception:  # noqa: BLE001
                call = None
        return text, ids, call

    def _normalize_paths(self, user_text: str) -> tuple[str, str | None]:
        """把用户话里的绝对路径归一化。

        项目内的 -> 改写成相对路径（沙箱只认相对路径）；
        项目外的 -> 返回 raw 让上层直接给提示，不让模型硬猜（它没见过绝对路径）。
        """
        outside: list[str] = []

        def repl(m: re.Match) -> str:
            raw = m.group(0).rstrip(" .")
            try:
                p = Path(raw).resolve()
            except (OSError, ValueError):
                outside.append(raw)
                return raw
            if p.is_relative_to(self.cwd):
                rel = p.relative_to(self.cwd).as_posix()
                return rel or "."
            outside.append(raw)
            return raw

        text = _ABS_IN_TEXT.sub(repl, user_text)
        return text, (outside[0] if outside else None)

    def _fallback_summary(self) -> str:
        """模型复读时的降级输出：直接给结构化结果，不糊半句垃圾。"""
        last = self.trace[-1]
        raw = filter_noise(last["result"])
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        if not lines:
            return "（没有输出）"
        # 优先给根目录层级的条目，子文件往后排，读者先看到主干
        shallow = [ln for ln in lines if ln.count("/") <= 1]
        deep = [ln for ln in lines if ln.count("/") > 1]
        picks = []
        for ln in shallow + deep:
            name = ln[2:] if ln.startswith("./") else ln
            if name == "." or name.startswith("."):
                continue
            picks.append(name)
            if len(picks) >= 8:
                break
        head = "、".join(ln[2:] if ln.startswith("./") else ln for ln in picks)
        more = "…" if len(lines) > len(picks) else ""
        return f"{last['tool']} 的结果（共 {len(lines)} 项）：{head}{more}"

    def step(self, user_text: str) -> str:
        user_text, outside = self._normalize_paths(user_text)
        # 护栏：小模型没见过绝对路径，遇到必然幻觉（选训练记忆里的路径 / 生成乱码）。
        # 与其让它硬猜，不如在壳层直接给出正确用法提示。
        if outside:
            return ("检测到你提到了项目外的绝对路径，我的沙箱只能在 --cwd 指定的目录里活动。\n"
                    f"请这样运行：chat.bat --cwd {outside}\n"
                    "然后再对我说\"看看这个项目的结构\"（话里不用再写路径）。")
        self.ids += self._enc(f"{U_TOKEN}{user_text}{END_TOKEN}")
        trace: list[dict] = []
        self.trace = trace
        for i in range(self.max_iters):
            self._truncate()
            self.ids += self._enc(A_TOKEN)
            t0 = time.time()

            text, ids, call = self._generate(self.ids, constrained=self.constrained,
                                             hint=user_text)
            if call is not None:
                name, args = call
                if validate_args(TOOL_MAP[name], args) is not None:
                    call = None
            if call is None and CALL_TOKEN in text and not self.constrained:
                # 自由生成解析失败，用语法约束再试一次（形状保证合法）
                text, ids, call = self._generate(self.ids, constrained=True, hint=user_text)
            text, looped = cut_phrase_loop(text)
            self.ids += ids
            if self.verbose:
                dt = time.time() - t0
                print(f"  [生成] {len(ids)} tok / {dt:.1f}s = {len(ids)/max(dt,1e-6):.1f} tok/s")

            if not call:
                if looped and trace:
                    # 复读说明它已经总结不动了：与其输出半句垃圾，不如给结构化结果
                    return self._fallback_summary()
                if looped or CALL_TOKEN in text:
                    # 无上下文可降级（如知识问答直接退化）或漏出调用原文：
                    # 小模型 OOD 的标准表现。清掉被污染的上下文，给一句能看的话。
                    self.ids = []
                    return ("抱歉，这条我没处理明白。我只会文件/目录操作和运行命令，"
                            "知识问答类的问题超出了我的能力；操作类的请换个具体说法再试。")
                clean = re.sub(r"<\|[^|]*\|>", "", text.replace(END_TOKEN, "")).strip()
                # 小模型 OOD 时会把自己上一轮的调用 JSON 尾巴抄进答案，
                # 形如 `目录里有 xxx","args":{"path":"build"}}` —— 把语法垃圾裁掉。
                if '"args"' in clean:
                    clean = re.split(r'",\s*"args"', clean)[0].rstrip('" ,')
                return clean if clean else "（模型没有给出内容）"

            name, args = call
            if self.verbose:
                print(f"  [调用] {name} {json.dumps(args, ensure_ascii=False)}")
            ok, result = self._execute(name, args)
            trace.append({"tool": name, "args": args, "ok": ok, "result": result[:400]})
            # 回灌给模型的观测要截断：训练时的工具输出都是百来字，
            # 真实项目 find 动辄几十行，超长输出会直接把小模型带进复读。
            fed = result if len(result) <= 400 else result[:400] + "…（输出过长已截断）"
            fed = filter_noise(fed)
            self.ids += self._enc(f"{T_TOKEN}{RESULT_TOKEN}{fed}{END_TOKEN}")
            if not ok:
                # 失败信息本身就是下一轮的输入，不在这里重试
                continue
        return "达到最大轮数仍未完成任务。最后一次结果：" + (trace[-1]["result"][:200] if trace else "无")


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="与训练好的小模型对话/执行命令")
    ap.add_argument("--ckpt", default="checkpoints/sft.pt")
    ap.add_argument("--tokenizer", default="tokenizer/tokenizer_v3")
    ap.add_argument("--cwd", default=".", help="允许操作的目录")
    ap.add_argument("--once", default=None, help="只跑一条指令")
    ap.add_argument("--yes", action="store_true", help="写操作不再逐条确认")
    ap.add_argument("--temp", type=float, default=0.3,
                    help="采样温度；工具操作建议 0~0.3，越高越容易选错工具")
    ap.add_argument("--constrained", action="store_true", help="强制语法约束解码")
    a = ap.parse_args()

    def confirm(cmd: str) -> bool:
        if a.yes:
            return True
        try:
            return input(f"  允许执行 `{cmd}` ? [y/N] ").strip().lower() in ("y", "yes")
        except EOFError:
            return False

    agent = Agent(a.ckpt, a.tokenizer, a.cwd, confirm=confirm, temperature=a.temp,
                  constrained=a.constrained)
    print(f"模型已加载，工作目录 {agent.cwd}，输入 exit 退出。")

    if a.once:
        print(agent.step(a.once))
        return

    while True:
        try:
            q = input("\n你> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q in ("exit", "quit", "q"):
            break
        if not q:
            continue
        print("助手>", agent.step(q))


if __name__ == "__main__":
    main()
