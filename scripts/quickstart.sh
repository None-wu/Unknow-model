#!/usr/bin/env bash
# 一键跑通全流程。默认 mini 档（3GB 语料，约 10 小时）。
#
#   bash scripts/quickstart.sh            # 正式复现 62M
#   bash scripts/quickstart.sh --smoke    # 10 分钟冒烟，只为验证代码能跑
#   bash scripts/quickstart.sh --resume   # 中断后继续预训练
set -euo pipefail

cd "$(dirname "$0")/.."
PY="${PY:-python}"
[ -x ".venv/bin/python" ] && PY=".venv/bin/python"
[ -x ".venv/Scripts/python.exe" ] && PY=".venv/Scripts/python.exe"

MODE="full"
RESUME=""
for arg in "$@"; do
  case "$arg" in
    --smoke) MODE="smoke" ;;
    --resume) RESUME="--resume" ;;
    *) echo "未知参数：$arg"; exit 1 ;;
  esac
done

if [ "$MODE" = "smoke" ]; then
  echo "=== 冒烟模式：样本语料 + 60 步预训练，预计 10 分钟 ==="
  "$PY" scripts/prepare_corpus.py --tier sample
  [ -f data/tool_sft.jsonl ] || "$PY" run.py synth --n 120 --workers 4
  "$PY" tools/extra_sft.py
  "$PY" run.py tokenize --mode both --target-tokens 3000000
  "$PY" run.py pretrain --arch 37m --steps 60 --out checkpoints/base_smoke.pt
  "$PY" run.py sft --base checkpoints/base_smoke.pt --epochs 1 --out checkpoints/sft_smoke.pt
  "$PY" run.py eval --ckpt checkpoints/sft_smoke.pt --n 20 || true
  echo "=== 冒烟完成：能跑通即可，指标不作数（模型只训了 60 步）==="
  exit 0
fi

echo "=== 完整复现：3GB 语料 + 786M token 预训练，预计 10-13 小时 ==="
"$PY" scripts/prepare_corpus.py --tier mini

[ -f data/tool_sft.jsonl ] || "$PY" run.py synth --n 21000
[ -f data/tool_sft_final.jsonl ] || "$PY" tools/extra_sft.py

"$PY" run.py tokenize --mode both --target-tokens 800000000
"$PY" run.py pretrain --arch 60m --steps 24000 --out checkpoints/base60.pt $RESUME
"$PY" run.py sft --base checkpoints/base60.pt --epochs 2 --out checkpoints/sft.pt
"$PY" run.py eval --n 200 --dump out/eval_errors.jsonl
"$PY" run.py probe --show-args
"$PY" acceptance_test.py || true

echo "=== 全部完成：checkpoints/sft.pt 就是 README 里那个模型 ==="
