# samples

仓库自带的少量样本，用于**无网络/低磁盘**环境下验证流程能跑通。正式训练请用 `scripts/prepare_corpus.py` 下载完整语料。

| 文件 | 内容 | 用途 |
|------|------|------|
| `corpus/pretrain_sample.jsonl` | 预训练语料前 2000 条（1.5MB） | `prepare_corpus.py --tier sample` 会把它铺成 `data/pretrain_t2t_mini.jsonl` |
| `corpus/sft_sample.jsonl` | 对话语料前 600 条（1.8MB） | 同上，铺成 `data/sft_t2t_mini.jsonl` |
| `tool_traces.jsonl` | 24 条合成轨迹（single/chain/refuse 各 8 条） | 直观看数据格式，不含训练用途 |

## 语料格式

预训练（`pretrain_sample.jsonl`）：

```json
{"text": "给我生成一首有关秋天的诗歌。秋日早晨，清风拂面。..."}
```

对话（`sft_sample.jsonl`）：

```json
{"conversations": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

工具轨迹（`tool_traces.jsonl`）：

```json
{"kind": "single", "events": [
   {"t": "user", "text": "看看 build 目录里有什么"},
   {"t": "call", "name": "list_dir", "args": {"path": "build"}, "output": "app.log\nsmall.log", "ok": true},
   {"t": "assistant", "text": "build 目录里有 app.log、small.log 两个文件。"}
 ], "meta": {...}}
```

`kind` 三种：`single`（单步）、`chain`（多步串联）、`refuse`（越界请求，直接拒绝）。

## 授权

`corpus/` 来自 [jingyaogong/minimind_dataset](https://huggingface.co/datasets/jingyaogong/minimind_dataset)（Apache-2.0 / CC-BY-NC-2.0），仅作格式示例与流程自检，请勿用于其他用途。`tool_traces.jsonl` 由本项目合成，MIT。
