# Unknow

**一个 62M 参数、从零训练、能在本地命令行里真正干活的中文 Agent 小模型。**

它不做通用问答，只专注一件事：把中文口语指令翻译成**结构化的工具调用**，交给沙箱执行，再把结果说人话回来。全部代码约 4900 行纯 PyTorch，单张 8GB 消费级显卡 10 小时可以完整复现。

```text
你> 这目录下都有啥
   [调用] list_dir {"path": "."}
   [执行] ls -1 -a .
目录里有 ./build、./data、./docs、./logs、./src、README.md…

你> 把 build 里超过 1MB 的日志删掉
   [调用] find_large {"path": "build", "min_mb": 1}
   [执行] find build -type f -size +1M ...
已删除 2 个超过 1MB 的日志文件。
```

> English: Unknow is a 62M-parameter Chinese tool-calling agent model trained from scratch (PyTorch only, 786M tokens pretrain + 21k synthesized tool trajectories). It maps natural-language commands to a closed DSL of 40 shell/file/python tools, executes them in a sandbox, and reports back. 786M tokens / 10h on a single RTX 4060 Laptop 8GB. See [docs/REPRODUCE.md](docs/REPRODUCE.md).

---

## 效果

留出集 200 条（同分布，指令模板相同、参数不同）与口语探针 24 条（**独立编写**的口语说法，训练时从未出现）：

| 指标 | 37M 版 | **62M 版（当前）** |
|------|--------|--------------------|
| 留出集：产出合法调用 | 100% | **100%** |
| 留出集：工具选择正确 | 99.5% | **100%** |
| 留出集：参数完全匹配 | 99.0% | **100%** |
| 口语探针：工具选择 | 75% | **100%（24/24）** |
| 口语探针：参数全对 | 71% | **96%（23/24）** |
| 端到端验收（22 项） | 19 通过 / 2 部分 / 1 失败 | **22 通过 / 0 / 0** |

同一套数据、同一套流程，只把 `dim 576→768`、`ffn 1536→2048`，口语探针从 75% 跳到 100%——**瓶颈是容量，不是训练量**。完整对比与逐项明细见 [docs/BENCHMARK.md](docs/BENCHMARK.md)。

生成速度：RTX 4060 Laptop 上 40–140 tok/s，一条单步工具任务约 1–3 秒。模型约 250MB（fp32），纯 CPU 也能跑。

---

## 它是什么 / 不是什么

**是**：一个可完整复现的"小模型 + 封闭工具集 + 沙箱执行"工程样本。价值在于**方法论**：怎样用 21k 条自带 ground truth 的合成轨迹，把 62M 模型的工具调用做到接近满分。

**不是**：通用助手。它不认识训练分布外的路径写法，不会推理复杂代码，遇到陌生项目的长输出会退化。它的 100% 只在"输入形态与训练一致"时成立——这一点在 [docs/PITFALLS.md](docs/PITFALLS.md) 里有血泪记录。

---

## 快速开始

### 0. 拿代码、装依赖

```bash
git clone https://github.com/None-wu/Unknow-model.git
cd Unknow-model

python -m venv .venv
.venv/Scripts/activate          # Windows；Linux/macOS 用 source .venv/bin/activate
pip install -r requirements.txt  # 需要 CUDA 版 torch 请按 pytorch.org 指引安装
```

### 1. 拿模型权重（二选一）

**A. 直接下载（推荐，只想用）**

从本仓库的 **GitHub Release** 下载 `unknow-62m.pt`（约 250MB），放到 `checkpoints/sft.pt`。

**B. 自己训（想复现）**

见下一节，或直接跑一键脚本：

```bash
powershell -ExecutionPolicy Bypass -File scripts\quickstart.ps1 -Smoke   # Windows 冒烟，10 分钟
bash scripts/quickstart.sh --smoke                                       # Linux/macOS
```

### 2. 开始用

```bash
# Windows：双击 chat.bat，或
chat.bat --cwd D:\my-project

# 任意平台
python run.py chat --cwd D:\my-project --temp 0
```

交互里输入指令即可，`exit` 退出。**写操作（删除、覆盖）默认逐条确认**，按 `y` 执行。

单条指令（脚本化）：

```bash
python run.py chat --once "在项目里搜一下 TODO" --cwd D:\my-project --temp 0
```

| 参数 | 作用 |
|------|------|
| `--cwd DIR` | 允许操作的目录（沙箱白名单），默认当前目录 |
| `--temp 0` | 温度。工具操作建议 0~0.3；0.6 会开始选错工具 |
| `--once TEXT` | 只跑一条就退出 |
| `--yes` | 跳过写操作确认（**陌生项目不要加**） |
| `--constrained` | 强制语法约束解码，JSON 必定合法但参数准确率略降 |
| `--ckpt PATH` | 换检查点 |

两条使用铁律：**路径交给 `--cwd`，话里不要写绝对路径**；**陌生项目不加 `--yes`**。

---

## 完整复现

从零到跑出上表所有数字，五步（详见 [docs/REPRODUCE.md](docs/REPRODUCE.md)）：

```bash
python scripts/prepare_corpus.py --tier mini   # 1. 下载语料并提取纯文本缓存（3GB）
python run.py synth --n 21000                  # 2. 合成工具轨迹（真实执行校验）
python tools/extra_sft.py                      # 3. 追加 369 条针对性样本
python run.py tokenize --mode both             # 4. 分词：786M token 流 + SFT 掩码样本
python run.py pretrain --arch 60m --steps 24000  # 5. 预训练（约 10 小时 / 8GB 显卡）
python run.py sft --base checkpoints/base60.pt --epochs 2
python run.py eval --n 200 && python run.py probe
```

只想先确认流程能跑通？用样本档，10 分钟：

```bash
python scripts/prepare_corpus.py --tier sample
python run.py synth --n 120
python tools/extra_sft.py
python run.py tokenize --mode both --target-tokens 3000000
python run.py pretrain --arch 37m --steps 60
python run.py sft --epochs 1
python run.py eval --n 20
```

### 硬件与耗时（实测）

| 阶段 | 硬件 | 耗时 |
|------|------|------|
| 语料下载 + 提取（mini） | — | 30–60 分钟 |
| 工具轨迹合成 21k 条 | 20 核 CPU | 约 15 分钟（12 进程，28 条/秒） |
| 分词 786M token | CPU | 约 35 分钟 |
| 预训练 24000 步 | RTX 4060 Laptop 8GB | **9–10 小时**（24.6k tok/s，5.5GB 显存） |
| SFT 2 epoch | 同上 | 约 20 分钟 |
| 评测 + 探针 | 同上 | 约 5 分钟 |

---

## 语料

**原始语料不随仓库分发**（约 25GB，且上游是 CC-BY-NC 授权）。仓库提供下载脚本 + 极小样本：

| 内容 | 位置 | 说明 |
|------|------|------|
| 上游数据集 | [jingyaogong/minimind_dataset](https://huggingface.co/datasets/jingyaogong/minimind_dataset) | 预训练 8.3GB + 14.1GB 对话，Apache-2.0 / CC-BY-NC-2.0 |
| 下载/提取 | `scripts/prepare_corpus.py --tier sample\|mini\|full` | 支持 `HF_ENDPOINT=https://hf-mirror.com` 镜像 |
| 语料样本 | `samples/corpus/*.jsonl` | 2.6MB，供无网络冒烟 |
| 工具轨迹（最终版） | `data/tool_sft_final.jsonl` | 21.5k 条，9.7MB，**训练真正用的语料** |
| 留出评测集 | `data/tool_eval.jsonl` | 511 条，独立种子生成 |
| 轨迹样本 | `samples/tool_traces.jsonl` | 24 条 single/chain/refuse 各 8 条 |

工具轨迹是**本项目自己合成**的（`tools/synth.py` + `tools/extra_sft.py`），MIT 授权，可自由使用。它每条都在临时沙箱里真跑过，stdout 是真实的，不是大模型编的。

---

## 设计要点

1. **封闭工具 DSL**：40 个工具（29 读 / 10 写 / 1 执行），每个参数强类型、带候选集。模型不允许自由发明命令——这是 62M 能可靠工作的前提。
2. **真实执行合成**：每条轨迹在临时沙箱真跑命令，拿真实 stdout 当标签，再用规则转成人话回复。数据自带 ground truth，不需要人标注。
3. **协议用独立 special token**：`<|tool_call|>` / `<|tool_result|>` / `<|end|>` 把"说话"和"发命令"切成两种模式，避免 JSON 和自然语言互相污染。
4. **壳层兜住模型做不到的事**：绝对路径归一化、短语循环截断、观测截断与噪声过滤、复读时结构化降级。小模型负责"意图→工具"，确定性逻辑交给代码。

展开见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。

---

## 仓库结构

```
model/model.py          62M MiniGPT：GQA + RoPE + RMSNorm + SwiGLU + tied embedding
tools/schema.py         40 工具 DSL + 7 个协议 token + 参数校验
tools/executor.py       常驻 bash worker、沙箱、路径拦截、夹具
tools/synth.py          轨迹合成：模板 × 文件池 × 真实执行
tools/extra_sft.py      针对性增量样本（自然语言→代码、长标识符、真实项目结构）
tokenizer/              16k BPE 词表（tokenizer_v3 已随仓库提供）
trainer/                data（分词/掩码）· pretrain · sft
agent/loop.py           Agent 主循环 + 壳层护栏
agent/decoder.py        语法约束解码（状态机）
agent/eval.py           留出集评测
agent/probe.py          24 条口语探针（独立编写）
acceptance_test.py      22 项端到端验收
scripts/                语料准备 · 评测集生成 · 模型导出
docs/                   复现步骤 · 架构 · 踩坑 · 评测明细
```

---

## 评测口径（避免误读）

- **留出集 200 条**：同分布。模板相同、文件与参数不同，且与训练集指令去重（`scripts/make_eval.py`）。衡量"指令→工具"映射的稳定性，**不代表真实泛化**。
- **口语探针 24 条**：`agent/probe.py` 里手写的说法，训练时从未出现。这才是真实泛化的指标，也是容量瓶颈暴露的地方（37M: 75% → 62M: 100%）。
- **参数匹配只比必填参数**：模型多给可选参数不算错。
- **约束解码默认关闭**：实测在强底座下，逐 token 强制字面量反而把参数全对率从 99.5% 拉到 92%，只在自由生成解析失败时兜底。

---

## 已知短板（诚实清单）

1. **抽象计算意图仍偏弱**：`用 python 算 2 的 10 次方` 工具选对了，但曾生成 `code: "10"`；补了 108 条"自然语言→代码"样本后才修好。
2. **长字符串保真度**：`ACCEPTANCE_MARK` 一度被写成 `AcCEPTANCE_MARK`（大小写丢失）。
3. **陌生项目长输出会退化**：真实项目 `find` 输出几千行，模型会开始抄片段并陷入短语循环。壳层已做截断 + 降级摘要兜住，但"理解"是没有的。
4. **绝对路径是分布外输入**：必须在壳层归一化或改写，直接喂给模型会幻觉出训练记忆里的路径。
5. **容量天花板**：62M 在 24 条探针上到 100%，但参数仍差 1 条；继续堆数据收益递减，下一步应扩到 100M+。

---

## 踩坑记录

这个项目 90% 的时间花在 debug 上，且几乎每个 bug 都"看起来像模型不行，实际是数据/代码错了"。14 个真实坑（SFT 标签双位移、`<|end|>` 漏进自由文本、int 参数无终止符、约束解码反而降准确率……）全部整理在 [docs/PITFALLS.md](docs/PITFALLS.md)——**想复现的人建议先读它，能省两天**。

---

## 文档

| 文档 | 内容 |
|------|------|
| [docs/REPRODUCE.md](docs/REPRODUCE.md) | 从零复现：环境、六阶段命令、耗时表、常见问题、自检清单 |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | 设计决策：模型、协议、工具 DSL、沙箱、合成、约束解码、壳层护栏 |
| [docs/PITFALLS.md](docs/PITFALLS.md) | 14 个真实 bug 的现象/根因/修法 + 调试方法论 |
| [docs/BENCHMARK.md](docs/BENCHMARK.md) | 全部评测数据、37M vs 62M、四轮迭代、口径说明 |
| [docs/PUBLISH.md](docs/PUBLISH.md) | 发布到 GitHub 的步骤与自检清单 |
| [samples/README.md](samples/README.md) | 语料与轨迹的格式说明 |

---

## 授权

代码 MIT。语料来自 MiniMind 数据集（Apache-2.0 / CC-BY-NC-2.0），本仓库不再分发原始语料；模型权重是上述语料的衍生成果，商用前请自行确认上游授权。详见 [LICENSE](LICENSE)。
