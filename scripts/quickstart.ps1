# 一键跑通全流程（Windows PowerShell）
#
#   powershell -ExecutionPolicy Bypass -File scripts\quickstart.ps1            # 正式复现 62M
#   powershell -ExecutionPolicy Bypass -File scripts\quickstart.ps1 -Smoke     # 10 分钟冒烟
#   powershell -ExecutionPolicy Bypass -File scripts\quickstart.ps1 -Resume    # 中断续跑
param(
    [switch]$Smoke,
    [switch]$Resume
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

$py = "python"
if (Test-Path ".venv\Scripts\python.exe") { $py = ".venv\Scripts\python.exe" }

function Step($msg) { Write-Host "`n=== $msg ===" -ForegroundColor Cyan }

if ($Smoke) {
    Step "冒烟模式：样本语料 + 60 步预训练，预计 10 分钟"
    & $py scripts/prepare_corpus.py --tier sample
    if (-not (Test-Path "data\tool_sft.jsonl")) { & $py run.py synth --n 120 --workers 4 }
    & $py tools/extra_sft.py
    & $py run.py tokenize --mode both --target-tokens 3000000
    & $py run.py pretrain --arch 37m --steps 60 --out checkpoints/base_smoke.pt
    & $py run.py sft --base checkpoints/base_smoke.pt --epochs 1 --out checkpoints/sft_smoke.pt
    & $py run.py eval --ckpt checkpoints/sft_smoke.pt --n 20
    Write-Host "`n冒烟完成：流程通了，指标不作数（模型只训了 60 步）" -ForegroundColor Green
    exit 0
}

Step "完整复现：3GB 语料 + 786M token 预训练，预计 10-13 小时"
& $py scripts/prepare_corpus.py --tier mini

if (-not (Test-Path "data\tool_sft.jsonl")) { & $py run.py synth --n 21000 }
if (-not (Test-Path "data\tool_sft_final.jsonl")) { & $py tools/extra_sft.py }

& $py run.py tokenize --mode both --target-tokens 800000000

$pre = @("run.py", "pretrain", "--arch", "60m", "--steps", "24000", "--out", "checkpoints/base60.pt")
if ($Resume) { $pre += "--resume" }
& $py @pre

& $py run.py sft --base checkpoints/base60.pt --epochs 2 --out checkpoints/sft.pt
& $py run.py eval --n 200 --dump out/eval_errors.jsonl
& $py run.py probe --show-args
& $py acceptance_test.py

Write-Host "`n全部完成：checkpoints/sft.pt 就是 README 里那个模型" -ForegroundColor Green
