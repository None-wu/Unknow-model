"""统一入口：把整条流水线串成一条命令。

    python run.py synth    --n 21000              # 1. 合成工具轨迹（真执行校验）
    python run.py tokenize --mode both            # 2. 分词化：预训练流 + SFT 样本
    python run.py pretrain --arch 60m --steps 24000  # 3. 预训练底座（37m / 60m）
    python run.py sft      --base checkpoints/base60.pt --epochs 2   # 4. 工具对齐
    python run.py chat                            # 5. 交互使用
    python run.py eval     --n 200                # 6. 评测工具选择与参数准确率
    python run.py probe                           # 7. 口语化探针（另写说法，查泛化）

每个子命令都把参数原样透传给对应模块，可以直接
    python run.py pretrain --help
查看该阶段的全部选项。
"""
from __future__ import annotations

import importlib
import sys

STAGES = {
    "synth": ("tools.synth", "合成工具调用轨迹（模板 + 真实执行校验）"),
    "tokenize": ("trainer.data", "分词化：预训练 token 流 + SFT 掩码样本"),
    "pretrain": ("trainer.pretrain", "预训练底座模型（--arch 37m|60m）"),
    "sft": ("trainer.sft", "SFT：工具调用 + 简单对话"),
    "chat": ("agent.loop", "交互式对话/执行命令"),
    "eval": ("agent.eval", "评测工具选择与参数准确率"),
    "probe": ("agent.probe", "口语化探针：换个说法还认不认得工具"),
}


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        print("可用阶段：")
        for k, (mod, desc) in STAGES.items():
            print(f"  {k:9s} {desc}  ({mod})")
        return
    stage = sys.argv[1]
    if stage not in STAGES:
        raise SystemExit(f"未知阶段 {stage}，可选：{', '.join(STAGES)}")
    mod_name = STAGES[stage][0]
    sys.argv = [f"{mod_name}"] + sys.argv[2:]
    mod = importlib.import_module(mod_name)
    mod.main()


if __name__ == "__main__":
    main()
