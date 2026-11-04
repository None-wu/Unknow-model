"""沙箱执行器：合成语料与推理时共用的命令执行层。

安全边界（合成几十万条轨迹会真跑几十万次命令，这一层不能出错）：
1. 每个样本一个临时沙箱目录，命令一律在该目录下以相对路径执行；
2. 命令必须是 schema 模板渲染出来的形态，执行前再走一遍 guard 拦截绝对路径/父目录/危险二进制；
3. 超时 5s + 输出截断 2KB；
4. 结束后整目录删除，不触碰仓库里的真实文件。

性能：git-bash 每次启动约 480ms，逐条 subprocess 会让 20 万条数据跑一整天。
这里用常驻 bash worker（--noprofile --norc）复用进程，单条命令降到 5~15ms。
"""
from __future__ import annotations

import os
import queue
import re
import shlex
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent


def _find_bash() -> str:
    for c in (
        shutil.which("bash"),
        r"C:\Program Files\Git\bin\bash.exe",
        r"C:\Program Files (x86)\Git\bin\bash.exe",
        r"C:\Program Files\Git\usr\bin\bash.exe",
    ):
        if c and Path(c).exists():
            return str(c)
    raise RuntimeError("未找到 bash，可执行文件无法运行")


BASH = _find_bash()

_BANNED = re.compile(
    r"\b(sudo|su|dd|mkfs\w*|fdisk|shutdown|reboot|chmod|chown|chattr|"
    r"curl|wget|ssh|scp|nc|netcat|eval|source|exec)\b"
)


def guard_command(cmd: str) -> str | None:
    """返回拒绝原因，None 表示放行。

    用 shlex 切词后逐 token 判断，而不是全文正则——
    正则会把 cut -d'/' 和 sed 's/a/b/' 里的斜杠误判成绝对路径。
    """
    if not cmd.strip():
        return "空命令"
    if "\n" in cmd or "\r" in cmd:
        return "命令含换行"
    try:
        tokens = shlex.split(cmd, posix=True)
    except ValueError:
        return "命令引号不完整"
    # posix 切词会把反斜杠当转义符吃掉（C:\Windows -> C:Windows），
    # Windows 绝对路径会因此绕过检查，所以再用原始空白切词扫一遍。
    raw_tokens = cmd.split()
    for tok in tokens + raw_tokens:
        if tok == "..":
            return "命令含父目录引用"
        if tok.startswith("../") or "/../" in tok or tok.endswith("/.."):
            return "命令含父目录引用"
        if tok.startswith("/") and not tok.startswith("/dev/null"):
            return f"命令含绝对路径: {tok}"
        if re.match(r"^[A-Za-z]:[\\/]", tok) or tok.startswith("\\\\"):
            return f"命令含 Windows 绝对路径: {tok}"
    m = _BANNED.search(cmd)
    if m:
        return f"命令含禁用二进制: {m.group(0)}"
    return None


@dataclass
class ExecResult:
    ok: bool
    output: str
    exit_code: int
    duration: float
    blocked: str | None = None


class BashWorker:
    """常驻 bash 进程，按行喂脚本、按标记收结果。

    stdout 交给独立线程读，主线程按 deadline 等待，超时就杀掉重建，
    避免一条卡死的命令拖垮整批合成。
    """

    def __init__(self) -> None:
        self._proc: subprocess.Popen | None = None
        self._buf: list[bytes] = []
        self._cv = threading.Condition()
        self._reader: threading.Thread | None = None

    def start(self) -> "BashWorker":
        env = {
            "PATH": os.environ.get("PATH", ""),
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "TZ": "CST-8",
            "PYTHONIOENCODING": "utf-8",
            "TERM": "dumb",
        }
        self._buf = []
        self._proc = subprocess.Popen(
            [BASH, "--noprofile", "--norc", "-s"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            bufsize=0,
        )
        self._reader = threading.Thread(target=self._pump, args=(self._proc,), daemon=True)
        self._reader.start()
        return self

    def _pump(self, proc: subprocess.Popen) -> None:
        fd = proc.stdout.fileno()  # type: ignore[union-attr]
        while True:
            try:
                chunk = os.read(fd, 65536)
            except OSError:
                break
            if not chunk:
                break
            with self._cv:
                self._buf.append(chunk)
                self._cv.notify_all()
        with self._cv:
            self._cv.notify_all()

    def _drain(self) -> bytes:
        with self._cv:
            data = b"".join(self._buf)
            self._buf = []
        return data

    def run(self, cmd: str, cwd: str | Path, timeout: float = 5.0, max_bytes: int = 2048) -> ExecResult:
        blocked = guard_command(cmd)
        if blocked:
            return ExecResult(False, f"[blocked] {blocked}", -1, 0.0, blocked)
        if self._proc is None or self._proc.poll() is not None:
            self.start()

        token = uuid.uuid4().hex
        marker = f"__END__{token}__"
        script = (
            f"cd {shlex.quote(str(cwd))} 2>/dev/null || true\n"
            f"{{\n{cmd}\n}} 2>&1 </dev/null\n"
            f"__rc=$?\n"
            f"printf '\\n{marker}%d\\n' \"$__rc\"\n"
        )
        t0 = time.time()
        assert self._proc and self._proc.stdin
        try:
            self._proc.stdin.write(script.encode("utf-8"))
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError):
            self._restart()
            return ExecResult(False, "[error] 执行进程已退出", -1, time.time() - t0)

        want = marker.encode()
        acc = b""
        # 先把喂命令之前残留的零散输出吃掉，避免污染本条结果
        acc += self._drain()
        deadline = t0 + timeout
        while True:
            idx = acc.find(want)
            if idx >= 0:
                break
            remain = deadline - time.time()
            if remain <= 0:
                self._restart()
                return ExecResult(False, f"[timeout] 执行超过 {timeout}s", -9, time.time() - t0)
            with self._cv:
                self._cv.wait(min(remain, 0.05))
            acc += self._drain()

        head, tail = acc[:idx], acc[idx + len(want):]
        rc_line, _, tail = tail.partition(b"\n")
        try:
            rc = int(rc_line.strip() or -1)
        except ValueError:
            rc = -1
        text = head.decode("utf-8", errors="replace").strip()
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        if len(text.encode("utf-8")) > max_bytes:
            text = text.encode("utf-8")[:max_bytes].decode("utf-8", errors="ignore") + "\n...（输出过长已截断）"
        if tail.strip():
            with self._cv:  # 归还未消费的尾巴
                self._buf.insert(0, tail)
        return ExecResult(rc == 0, text, rc, time.time() - t0)

    def _restart(self) -> None:
        self.close()
        self.start()

    def close(self) -> None:
        if self._proc:
            try:
                self._proc.kill()
            except Exception:
                pass
            self._proc = None


# ---------------------------------------------------------------------------
# 沙箱文件系统
# ---------------------------------------------------------------------------
FIXTURE: dict[str, str] = {
    "README.md": "# 项目说明\n\n这是一个用于测试命令行的示例项目。\n运行 `bash run.sh` 开始。\n",
    "run.sh": "#!/bin/bash\necho start\necho done\n",
    "build/app.log": "".join(f"2026-10-{i % 28 + 1:02d} INFO build step {i} ok\n" for i in range(120)),
    "build/small.log": "warn: deprecated api\n",
    "build/notes.txt": "TODO: 补测试\nTODO: 优化启动速度\n",
    "build/old.bak": "backup placeholder\n",
    "src/main.py": "def main():\n    print('hello')\n\n\nif __name__ == '__main__':\n    main()\n",
    "src/utils.py": "def add(a, b):\n    return a + b\n\n\ndef mul(a, b):\n    return a * b\n",
    "src/config.json": '{"name": "demo", "version": "1.0.0", "debug": true}\n',
    "data/users.csv": "id,name,city\n1,alice,beijing\n2,bob,shanghai\n3,carol,shenzhen\n",
    "data/orders.csv": "oid,uid,amount\n1001,1,29.9\n1002,2,88.0\n1003,1,12.5\n",
    "data/empty.txt": "",
    "logs/access.log": "GET / 200\nGET /api 200\nPOST /api 500\nGET / 200\nGET /favicon.ico 404\n",
    "logs/error.log": "ERROR db timeout\nERROR retry failed\nWARN slow query\n",
    "docs/guide.md": "# 指南\n\n第一步：安装依赖。\n第二步：运行测试。\n",
    "docs/changelog.md": "## 1.0.0\n- 初始版本\n## 1.1.0\n- 修复登录问题\n",
    "tmp/cache.bin": "x" * 1024,
    "data/events.jsonl": '{"id":1,"op":"view"}\n{"id":2,"op":"click"}\n{"id":3,"op":"buy"}\n',
}
BIG_FILE = "data/dump.bin"
BIG_MB = 3


def fixture_template() -> Path:
    """模板目录只建一次；大文件单独生成，其余都是小文本。"""
    root = _REPO / "tools" / "_fixture"
    if (root / ".ready").exists():
        return root
    for rel, content in FIXTURE.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    big = root / BIG_FILE
    big.parent.mkdir(parents=True, exist_ok=True)
    with open(big, "wb") as f:
        f.write(b"\0" * (BIG_MB * 1024 * 1024))
    (root / ".ready").write_text("1")
    return root


def list_paths(root: str | Path, max_n: int = 200) -> list[str]:
    """列出目录下的相对路径，作为路径参数的候选集。"""
    root = Path(root)
    out = ["."]
    try:
        for p in sorted(root.rglob("*")):
            rel = p.relative_to(root).as_posix()
            if rel.startswith(".") or "__pycache__" in rel:
                continue
            out.append(rel)
            if len(out) >= max_n:
                break
    except OSError:
        pass
    return out


_MENTION = re.compile(r"[A-Za-z0-9_\-]+(?:/[A-Za-z0-9_.\-]+)+"
                      r"|\b[A-Za-z0-9_\-]+\.(?:py|txt|md|json|jsonl|csv|log|bin|yaml|yml|toml|sh|ini|bak)\b")


def mention_paths(text: str, max_n: int = 12) -> list[str]:
    """从用户话里抠出提到的路径。

    路径参数是枚举的（从环境清单里选），但用户可能点名一个当前不存在的文件
    ——比如"把 tmp/x.bin 清空"。这时清单里没有它，模型只能被逼着选个错的。
    把话里出现的路径并进候选集，模型就只需要做"选哪个"，而不是"凭空拼路径"；
    选中的路径若真的不存在，执行层会如实报错，这比悄悄改目标诚实得多。
    """
    out: list[str] = []
    for m in _MENTION.finditer(text):
        s = m.group(0).rstrip(".,:;，。：；")
        if s and s not in out:
            out.append(s)
    return out[:max_n]


def candidates_for(root: str | Path, text: str = "", max_n: int = 200) -> list[str]:
    """环境清单 + 话里提到的路径，去重后作为路径参数的候选集。"""
    merged = list(dict.fromkeys(mention_paths(text) + list_paths(root, max_n)))
    return merged


def fixture_paths() -> list[str]:
    """与 FIXTURE 一致的候选清单，供评测/离线场景使用。"""
    paths = {"."}
    for rel in list(FIXTURE) + [BIG_FILE]:
        parts = rel.split("/")
        for i in range(1, len(parts)):
            paths.add("/".join(parts[:i]))
        paths.add(rel)
    return sorted(paths)


def run(cmd: str, cwd: str | Path, timeout: float = 5.0) -> ExecResult:
    """一次性执行（不复用进程）。批量场景请直接复用 BashWorker。"""
    w = BashWorker().start()
    try:
        return w.run(cmd, cwd, timeout=timeout)
    finally:
        w.close()


class Sandbox:
    """with Sandbox(worker) as sb: sb.run(cmd)

    小文件逐个复制（约 10KB），大文件用硬链接——秒级创建且删除互不影响。
    reuse=True 时目录只建一次、退出不删，专供只读命令复用，
    避免在 Windows 上每样本重建 ~19 个文件被实时防护反复扫描。
    """

    def __init__(self, worker: BashWorker, base: Path | None = None, reuse: bool = False):
        self.template = fixture_template()
        self.worker = worker
        self.root: Path | None = None
        self._base = base
        self.reuse = reuse

    def _populate(self, dst: Path) -> None:
        for src in self.template.rglob("*"):
            rel = src.relative_to(self.template)
            if src.is_dir() or rel.name == ".ready" or str(rel) == BIG_FILE:
                continue
            target = dst / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, target)
        big_src = self.template / BIG_FILE
        if big_src.exists():
            try:
                os.link(big_src, dst / BIG_FILE)
            except OSError:
                shutil.copyfile(big_src, dst / BIG_FILE)

    def __enter__(self) -> "Sandbox":
        if self.reuse and self.root is not None and self.root.exists():
            return self
        self.root = Path(tempfile.mkdtemp(prefix="sbx_", dir=self._base))
        self._populate(self.root)
        return self

    def __exit__(self, *exc) -> None:
        if self.reuse:
            return
        if self.root and self.root.exists():
            shutil.rmtree(self.root, ignore_errors=True)
            self.root = None

    def destroy(self) -> None:
        if self.root and self.root.exists():
            shutil.rmtree(self.root, ignore_errors=True)
        self.root = None

    def run(self, cmd: str, timeout: float = 5.0) -> ExecResult:
        assert self.root is not None
        return self.worker.run(cmd, self.root, timeout=timeout)


if __name__ == "__main__":
    from tools.schema import TOOLS, render_command

    samples = [
        ("list_dir", {"path": "build"}),
        ("grep_text", {"pattern": "TODO", "path": "."}),
        ("find_large", {"path": ".", "min_mb": 2}),
        ("du_dir", {"path": "build"}),
        ("cut_columns", {"path": "data/users.csv", "sep": ",", "col": 2}),
        ("python_run", {"code": "print(sum(range(10)))"}),
        ("date_now", {}),
        ("replace_text", {"path": "build/notes.txt", "old": "TODO", "new": "DONE"}),
        ("find_ext", {"path": ".", "ext": "py"}),
    ]
    w = BashWorker().start()
    try:
        t0 = time.time()
        n = 0
        with Sandbox(w) as sb:
            for name, args in samples:
                cmd = render_command(next(t for t in TOOLS if t.name == name), args)
                r = sb.run(cmd)
                n += 1
                print(f"[{name}] rc={r.exit_code} {r.duration*1000:.0f}ms :: {cmd}")
                for line in r.output.splitlines()[:3]:
                    print("    ", line[:90])
        print(f"\n{len(samples)} 条命令总耗时 {time.time()-t0:.2f}s")

        # 冷启动/热路径对比：连续 200 条 list_dir
        t1 = time.time()
        with Sandbox(w) as sb:
            for _ in range(200):
                sb.run("ls -1 build")
        print(f"200 条 list_dir: {time.time()-t1:.2f}s（单条 {(time.time()-t1)/200*1000:.1f}ms）")

        # 安全闸
        with Sandbox(w) as sb:
            print("逃逸尝试:", sb.run("cat ../../etc/passwd").output)
            print("绝对路径:", sb.run("ls /etc").output)
            print("危险命令:", sb.run("sudo rm -rf .").output)
    finally:
        w.close()
