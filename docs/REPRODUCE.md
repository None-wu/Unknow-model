# 完整复现指南

从一台干净的机器，到跑出 README 里那张表的全部数字。

预计总耗时：**10–13 小时**（含语料下载与提取），其中 9–10 小时是预训练，可以挂着睡觉。

---

## 0. 环境

| 项 | 要求 |
|----|------|
| Python | ≥ 3.10（本项目在 3.10.9 上开发） |
| 内存 | ≥ 16GB |
| 显存 | ≥ 6GB（62M 预训练实测峰值 5.5GB）；纯 CPU 也能跑，慢 30 倍以上 |
| 磁盘 | `mini` 档约 12GB，`full` 档约 60GB |
| 实测硬件 | RTX 4060 Laptop 8GB / 16GB RAM / 20 核 |

```bash
git clone <your-repo> unknow && cd unknow
python -m venv .venv
.venv/Scripts/activate            # Windows
# source .venv/bin/activate       # Linux/macOS
pip install -r requirements.txt
```

验证环境：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

---

## 1. 准备语料

原始语料约 25GB，不随仓库分发。三档任选：

```bash
# 冒烟：用仓库自带样本（2.6MB），10 分钟跑完全流程，只为验证代码能跑
python scripts/prepare_corpus.py --tier sample

# 正式复现：3.0GB 下载，能复现 786M token 的预训练
python scripts/prepare_corpus.py --tier mini

# 完整：24GB 下载，与本文所有数字一致（mini 档已足够，full 只是语料更全）
python scripts/prepare_corpus.py --tier full
```

国内网络建议先设镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com     # Windows: set HF_ENDPOINT=https://hf-mirror.com
```

产物：

```
data/pretrain_t2t_mini.jsonl        1.2 GB   预训练语料
data/sft_t2t_mini.jsonl             1.7 GB   对话语料（SFT 混训）
data/lora_medical.jsonl              34 MB   医学对话（可选）
_corpus_cache/*_extracted.txt       ~2.3 GB   一行一条的纯文本，分词与词表训练都读它
```

> 为什么要有 `_corpus_cache`：分词阶段每 6 字节估一个 token，直接读 JSON 会反复解析 GB 级文件；先摊平成纯文本，分词只要流式读行。

---

## 2. 合成工具轨迹

```bash
python run.py synth --n 21000 --workers 12
```

- 每条轨迹在临时沙箱里**真跑**命令，stdout 就是标签，不需要人工标注
- 三种样本：`single`（单步）、`chain`（2–5 步串联，后一步参数依赖前一步输出）、`refuse`（越界请求直接拒绝）
- 实测 28 条/秒（12 进程），21k 条约 15 分钟，通过率约 85%（失败的轨迹会被丢弃）

产出 `data/tool_sft.jsonl`（约 21k 条 / 9.6MB）。

然后追加针对性样本：

```bash
python tools/extra_sft.py --src data/tool_sft.jsonl --out data/tool_sft_final.jsonl
```

这 369 条解决的是标准合成数据覆盖不到的四类问题（每条都对应一个实测失败）：

| 类别 | 条数 | 解决什么 |
|------|------|----------|
| 自然语言 → 代码 | 108 | `算一下 2 的 10 次方` 只吐 `10` 而非 `print(2**10)` |
| cut_columns 中文分隔符 | 54 | `按逗号分隔` 不会映射成 `,` |
| 长标识符保真 | 16 | `ACCEPTANCE_MARK` 大小写丢失 |
| 身份对话 ×8 | 112 | `你是谁` 被误判为 `whoami` 调用 |
| 真实项目结构摘要 | 42 | 陌生项目的 `tree` 输出会引发复读 |
| list_dir 口语说法 | 20 | `里面有些什么文件` 误选 `find_ext` |

> 这段是可选的：只用 `data/tool_sft.jsonl` 也能训练，但 python_run 与口语探针会明显掉分。

---

## 3. 分词

```bash
python run.py tokenize --mode both --target-tokens 800000000 --tool-data data/tool_sft_final.jsonl
```

产出：

```
data/tokenized/pretrain.bin   1.5GB   预训练 token 流（uint16，786M token）
data/tokenized/sft.pt          28MB   SFT 样本（ids + 掩码后的 labels）
```

- `pretrain.bin` 从 `_corpus_cache` 的多份纯文本按**字节数配额**等比抽样（等距跳行，不从头顺读），
  每条后接 `</s>`，编码成 uint16 流；token 量按 6 字节/token 保守估算，抽不满会自动放大目标重跑
- `sft.pt` 是 21.5k 轨迹 + 8.4k 通用对话（`--chat-ratio 0.28`），**只对 assistant 段计 loss**
- 分词结束会打印一条还原样本，**务必肉眼确认**协议 token 与掩码是否正确（这一步能提前发现 80% 的坑）

词表已随仓库提供（`tokenizer/tokenizer_v3`，16384 词，1.90 字/token，往返 100%）。想自己训：

```bash
python -m tokenizer.sample_corpus            # 从 _corpus_cache 按 45/40/15 混出 320MB
python tokenizer/build_vocab.py --vocab-size 16384
```

---

## 4. 预训练

```bash
# 62M（推荐，本文数字来自这里）
python run.py pretrain --arch 60m --steps 24000 --out checkpoints/base60.pt

# 37M（更快，跑 6.5 小时，探针只能到 75%）
python run.py pretrain --arch 37m --steps 24000 --out checkpoints/base.pt
```

| 架构 | 参数 | dim | 层 | ffn | 显存 | 速度 | 耗时 |
|------|------|-----|----|-----|------|------|------|
| `37m` | 37.3M | 576 | 8 | 1536 | 4.4GB | 33.5k tok/s | ~6.5h |
| `60m` | 62.1M | 768 | 8 | 2048 | 5.5GB | 24.6k tok/s | ~9.5h |

24000 步 × batch 16 × accum 4 × seq 512 = **786M token**。相对 full 档全部语料这不到一遍（约 1/5），
mini 档则会重复约 1.3 遍——62M 模型在这个量级已收敛，再堆 token 的收益远小于扩容。

预期日志：

```
step 24000/24000 | loss 2.176 | lr 3.0e-05 | ppl 8.8 | 24.6k tok/s | 已用 9h31m
```

中断不要紧，每 1000 步存档，`--resume` 续跑：

```bash
python run.py pretrain --arch 60m --steps 24000 --out checkpoints/base60.pt --resume
```

---

## 5. SFT

```bash
python run.py sft --base checkpoints/base60.pt --epochs 2 --out checkpoints/sft60.pt
```

约 20 分钟，验证 loss 应降到 **0.99 附近**（37M 版是 1.066）。

```bash
python run.py sft --base checkpoints/sft60.pt --epochs 2 --out checkpoints/sft60_v2.pt   # 可选：再训 2 轮
```

> 如果 SFT 后生成结果是 `{"":topf",args{...}` 这种跳字乱码，**不是模型问题，是标签位移了两次**——见 [PITFALLS.md 第 2 条](PITFALLS.md)。

---

## 6. 评测

```bash
python run.py eval --n 200 --dump out/eval_errors.jsonl    # 留出集
python run.py probe --show-args                            # 口语探针
python acceptance_test.py                                  # 22 项端到端验收
```

预期（62M 模型）：

```
留出集 200 条：出调用 100% | 完整合法 100% | 工具选择 100% | 参数匹配 100% | 沙箱可执行 100%
口语探针 24 条：工具选择 100% (24/24) | 参数全对 96% (23/24)
验收 22 项：22 通过 / 0 部分 / 0 失败
```

### 评测集是哪来的

`data/tool_eval.jsonl` 已随仓库提供（511 条）。要自己重新生成（不同种子 + 剔除训练集重复指令）：

```bash
python scripts/make_eval.py --n 500 --out data/tool_eval.jsonl
```

⚠️ 如果你用 `run.py synth` 的同一份输出既当训练又当评测，指标会虚高——这是"同源评测"，没有意义。

---

## 7. 打包发布

```bash
python scripts/export_model.py --ckpt checkpoints/sft.pt --out release/unknow-62m.pt
```

712MB → 约 250MB（剥掉优化器状态）。GitHub Release 单文件上限 2GB，直接传即可；若想放进 git 仓库需配 Git LFS。

---

## 常见问题

| 现象 | 原因 | 处理 |
|------|------|------|
| `ModuleNotFoundError: torch` | 用了系统 python 而不是 venv | 统一用 `.venv/Scripts/python.exe`，或激活后跑 |
| 下载卡住 / 超时 | HF 直连不稳 | `export HF_ENDPOINT=https://hf-mirror.com` 重试 |
| 预训练 OOM | 显存 < 6GB | 降 `--batch 8 --accum 8`（等效 batch 不变），或 `--seq-len 256` |
| 分词产出远少于 786M | 抽样步长估偏 | `train_all.py` 会自动放大目标重跑；手动跑就把 `--target-tokens` 调大 |
| 生成乱码跳字 | 标签位移两次 | 见 [PITFALLS.md 第 2 条](PITFALLS.md) |
| 生成的 JSON 尾巴漏进回答 | `<|end|>` 不在 special_ids | 见 [PITFALLS.md 第 5 条](PITFALLS.md) |
| 中文里夹 `Eump` 之类乱码 | 分布外输入导致短语循环 | 壳层已截断；根治靠数据多样性 |
| 评测参数匹配偏低 | 用约束解码了 | 去掉 `--constrained`，实测自由生成更准 |

---

## 复现检查清单

- [ ] `pip install -r requirements.txt`，`torch.cuda.is_available()` 为 True
- [ ] `prepare_corpus.py` 产出 `_corpus_cache` 四个 txt（mini 档约 2.3GB）
- [ ] `synth` 通过率 ≥ 80%，产出 ≥ 18k 条
- [ ] `extra_sft.py` 输出 `21518+369` 量级
- [ ] 分词后还原样本里 `<|user|>…<|assistant|><|tool_call|>{…}` 结构完整
- [ ] 预训练 loss 收在 2.2 附近，ppl < 10
- [ ] SFT 验证 loss < 1.05
- [ ] `eval` 五项中前三项 100%
- [ ] `probe` 工具选择 ≥ 95%
- [ ] `acceptance_test.py` ≥ 20 项通过
