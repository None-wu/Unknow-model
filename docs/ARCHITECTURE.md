# 架构与设计决策

这个项目的每个设计选择，都是为了回答一个问题：**怎样让 62M 参数的模型可靠地操作命令行？**

答案是四条：**封闭的动作空间、自带 ground truth 的数据、把"说话"和"发命令"分成两种模式、把模型做不到的确定性逻辑挪到壳层**。

---

## 1. 模型

纯 PyTorch 手写，268 行（`model/model.py`），不依赖 transformers 的模型代码。

| 项 | 62M 版 | 37M 版 |
|----|--------|--------|
| 参数 | 62.14M | 37.32M |
| 宽度 dim | 768 | 576 |
| 层数 | 8 | 8 |
| 注意力 | 8 头 / 2 KV 头（GQA） | 同 |
| FFN | 2048（SwiGLU） | 1536 |
| 词表 | 16384 | 同 |
| 上下文 | 1024 | 同 |
| 位置编码 | RoPE（θ=10000） | 同 |
| 归一化 | RMSNorm（eps 1e-5） | 同 |
| 输出层 | 与 embedding 共享权重 | 同 |

单卡 8GB 上 62M 模型预训练峰值 **5.5GB**（batch 16 × accum 4 × seq 512，bf16 自混精度）。

**为什么是 62M 而不是 100M+**：目标场景是"单条命令、单步或两三步串联"，不需要世界知识。62M 是"探针 100% 达成"与"笔记本 10 小时能训完"的交点。再往上，收益要多花一倍时间才看得到。

**容量是硬约束，这是实测结论**：37M 在 24 条口语探针上卡在 75%，堆数据、加同义改写都救不回来（同义改写甚至把 75% 拉到 71%，因为容量被多样表达稀释）。只把 dim 576→768，同一份数据，探针直接 100%。

---

## 2. 对话协议

```
<|user|>把 build 里超过 1MB 的日志删掉<|end|>
<|assistant|><|tool_call|>{"name":"find_large","args":{"path":"build","min_mb":1}}<|end|>
<|tool|><|tool_result|>build/a.log
build/b.log<|end|>
<|assistant|><|tool_call|>{"name":"delete_file","args":{"path":"build/a.log"}}<|end|>
<|tool|><|tool_result|>deleted build/a.log<|end|>
<|assistant|>已删除 2 个超过 1MB 的日志文件。<|end|>
```

7 个协议 token（`<|user|> <|assistant|> <|tool|> <|tool_call|> <|tool_result|> <|end|> <|think|>`）硬编码进词表，id 固定。

**为什么要独立 token，而不是纯 JSON**：小模型很容易在自由文本里"顺便"写出 `{"name":...}`，或者把 JSON 写进回答。用不可打印的 special token 划出硬边界，模型在"回答模式"下不会突然发命令——这条在 62M 上比在 7B 上重要得多。

**为什么 assistant 段可以出现多次**：一次任务的多轮工具调用都发生在同一个 assistant 回合里（OpenAI 的 tool 调用也是这个结构）。这样模型学会的是"拿到结果→决定下一步"，而不是"每轮重新开始"。

---

## 3. 工具 DSL（`tools/schema.py`）

40 个工具，按安全等级分三类：

| 等级 | 数量 | 例 |
|------|------|-----|
| `read` | 29 | `list_dir` `tree` `cat_file` `grep_text` `find_large` `cut_columns` `diff_files` `disk_free` `which_cmd` |
| `write` | 10 | `mkdir` `write_file` `append_file` `copy_file` `move_file` `delete_file` `delete_dir` `replace_text` |
| `exec` | 1 | `python_run` |

每个工具是 `Tool(name, desc, args, template, intents, level, emits_paths, tags)`：

- `args`：强类型（`path` / `text` / `int` / `bool` / `choice` / `pattern` / `cmd`），可带候选集、上下界
- `template`：结构化调用 → POSIX 命令的渲染模板，例如 `ls -1 -a {path}`
- `intents`：训练时用来扩写用户指令的模板
- `emits_paths`：标记"这个工具的输出是路径列表"，用于生成 chain 样本时把上一步输出喂给下一步

**封闭集合是核心约束**。模型不能发明 `curl | bash`，因为工具表里没有。这比任何"安全提示词"都可靠——62M 模型根本记不住安全规则，但它能学会 40 个名字。

---

## 4. 执行与沙箱（`tools/executor.py`）

```
Agent ──► Sandbox ──► BashWorker（常驻进程）──► bash -c "命令"
              │
              ├─ 路径拦截（guard_command）
              ├─ 硬链接夹具（大文件不复制）
              └─ 超时/输出上限
```

- **常驻 bash worker**：每条命令 56ms，而 `subprocess.run` 每次新起进程是 480ms。合成 21k 条轨迹要跑 10 万+ 命令，这个差别是 15 分钟 vs 2 小时。
- **沙箱**：每个样本一个临时目录，需要的文件用**硬链接**放进去（`BIG_FILE` 2MB 不真的复制）。
- **路径白名单**：`guard_command` 拦绝对路径、`..` 父目录、Windows 盘符路径。注意这里有个真实漏洞：`shlex.split(posix=True)` 会把 `C:\Windows\System32` 的反斜杠当转义符吃掉，切成 `C:WindowsSystem32` 绕过检查——现在额外用原始空白切词二次扫描（见 PITFALLS 第 9 条）。
- **写操作确认**：`write` 级工具默认逐条询问，`--yes` 可跳过。

---

## 5. 数据合成（`tools/synth.py`）

不调大模型编数据，原因很直接：**编出来的 stdout 是假的**，模型会学到"命令和输出之间的虚假关联"，比如以为 `wc -l` 的输出总是 `42`。

这里的做法：

```
意图模板 × 文件池轮转 × 真实执行 → 通过校验才入库
```

1. 从工具的 `intents` 里抽一句，填上随机文件/目录/参数
2. 渲染成命令，在沙箱真跑
3. **只保留执行成功、且输出可被规则解释的轨迹**
4. 用规则把 stdout 转成人话回答（`disk_free` → "磁盘剩余 47G，已用 85%"）

产出三种样本：

| 类型 | 内容 | 作用 |
|------|------|------|
| `single` | 一句话 → 一个调用 → 一个回答 | 基础映射 |
| `chain` | 2–5 步，后一步参数依赖前一步输出 | 多轮决策 |
| `refuse` | 越界请求（删系统目录、看别的用户文件）→ 不出工具直接拒绝 | 安全边界 |

**文件池轮转**解决的是数据耦合：最开始 `truncate_file` 只见过 `.log`，`find_name` 总是接到内容关键词当文件名（`find . -name TODO`）。让文件池按工具语义轮转（截断工具轮 `logs/bak/txt`，查找工具轮各类名字）之后才干净。

**关键认知**：小模型的工具能力主要来自**这部分合成轨迹 + 执行壳提供的环境信息**，而不是预训练。同样 786M token 的底座，不加 SFT 完全不会发调用；加了 21k 条，参数匹配就上 99%。

---

## 6. 约束解码与它的反直觉结论（`agent/decoder.py`）

`ConstrainedDecoder` 是一个状态机：把生成过程切成 `lit`（字面量）/`tool`（工具名枚举）/`enum`/`int`/`free` 步骤，每步只允许合法 token 通过 logits 掩码，保证输出 100% 是合法 JSON 且工具名一定在表里。

听起来应该稳赢，**实测相反**：

| 方案 | 参数全对率 |
|------|-----------|
| 自由生成 + schema 校验 | **99.5%** |
| 语法约束解码 | 92.0% |

原因：逐 token 强制字面量，会破坏模型原生的 tokenization。模型本来想输出一个多字符 token（`"build/app.log"`），被拆成逐字符后，它在每个位置看到的上下文都是"没见过的形状"，概率分布被切碎，反而更容易在参数内容上出错。

所以现在的策略是**自由生成为主，约束解码兜底**：

```
自由生成 → 解析 → schema 校验
   ├─ 成功 → 执行
   └─ 失败 → 用约束解码重生成一次（形状必定合法）
```

`--constrained` 保留，用于极端场景（比如自由生成反复解析失败）。这是本项目最有价值的一条实测结论：**结构约束不是免费的**。

---

## 7. Agent 主循环与壳层护栏（`agent/loop.py`）

```
user_text
  ├─ [护栏] 绝对路径？→ 项目内改写为相对路径；项目外直接返回 --cwd 用法提示（不进模型）
  ├─ 拼 prompt → 生成
  ├─ 有 tool_call？→ schema 校验 → 沙箱执行 → 结果回灌 → 继续生成（最多 N 轮）
  └─ 没有 → 清理答案 → 返回
```

壳层还做了四件模型做不到的事：

| 护栏 | 解决什么 |
|------|----------|
| 绝对路径归一化 | 模型没见过 `E:\xxx` 这种形态，会幻觉出训练记忆里的路径 |
| 短语循环截断（3–48 字重复 ≥3 次） | `repeat_stop` 只防单 token 连续重复，防不住"目录里有 Eump，"整句循环 |
| 观测截断到 400 字 + 滤 `.git`/`node_modules`/`dist` | 真实项目输出几千行，模型会开始抄片段 |
| 复读时降级为结构化摘要 | 与其吐半句垃圾，不如直接给 `tree 的结果（共 14 项）：…` |

这不是"绕过模型缺陷"，而是**分工**：模型负责它擅长的语义映射，确定性逻辑用代码写死。62M 模型不可能学会"路径在 cwd 外要提示用户"，但三行代码可以。

---

## 8. 分词（`tokenizer/`）

byte-level BPE，16384 词表，7 个协议 token 固定 id。

- 实测 **1.90 字/token**，往返（encode→decode）100%
- 语料配比：通用 45% / 对话 40% / 工具轨迹 15%——工具轨迹占比要够高，否则命令行语法在词表里权重不足，`grep -rn` 会被切得很碎
- 词表用 320MB 混合文本训练就收敛，不需要全量 19GB

**踩过的坑**：在 Git Bash 里传 `--special-tokens '</s>'`，MSYS 会把 `</s>` 当路径转成 `<C:/.../s>` 塞进词表，导致 eos 缺失。所以有 `tokenizer/build_vocab.py`——协议 token 硬编码在脚本里，不经过 shell。

---

## 9. 评测设计

三层，各测一件事：

| 层 | 数据 | 测什么 | 局限 |
|----|------|--------|------|
| 留出集 | `tool_eval.jsonl` 511 条，独立种子 + 去重 | 指令→工具映射的稳定性 | 同分布，指标会偏高 |
| 口语探针 | `probe.py` 24 条手写说法 | 真实泛化 | 样本少，1 条 = 4pp |
| 端到端验收 | `acceptance_test.py` 22 项 | 安全、闭环、边界、鲁棒性 | 人工设计，覆盖有限 |

**参数匹配只比必填参数**：模型多给一个可选参数（比如 `read_head` 顺手给了 `lines`）不算错——这是我们早期误判的来源。

**为什么必须有三层**：留出集 100% 的时候，探针只有 75%。如果只看留出集，会得出"模型已经可用"的错误结论。真实泛化只有换说法才测得出来。

---

## 10. 目录职责

```
model/        纯模型定义，无训练逻辑
tools/        schema（工具定义）+ executor（执行/沙箱）+ synth（数据合成）
trainer/      data（分词与掩码）+ pretrain + sft + utils
agent/        loop（主循环/护栏）+ decoder（约束解码）+ eval + probe
tokenizer/    词表训练
scripts/      语料准备、评测集生成、模型导出
```

依赖方向单向：`agent → tools → model`，`trainer → model/tools`。没有循环依赖，任何一层都可以单独跑测试。
