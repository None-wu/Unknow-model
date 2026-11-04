# 发布到 GitHub

仓库已经整理成可直接推送的状态。以下是操作步骤。

---

## 1. 确认当前状态

```bash
git branch            # 应该只有 main（以及保留的 main-legacy）
git log --oneline     # 只有一条提交
git status            # 干净
```

- `main`：干净历史，**47 个文件 / 14.5MB**，`git bundle` 实测推送体积约 **2.4MB**
- `main-legacy`：旧历史（含 5.5GB 的 `.venv`，54504 个文件），**仅供本地留档，永远不要推**

## 2. 建远程仓库并推送

已发布在 **https://github.com/None-wu/Unknow-model**（public，含 topics 标签）。

```bash
git remote add origin https://github.com/None-wu/Unknow-model.git
git push -u origin main
```

推送完成后，GitHub 会以 `main` 为默认分支，历史里不会出现任何 `.venv` 文件。

> 如果 `git push` 提示 `main-legacy` 也要推，说明用了 `--all`。改成只推 `main` 即可。

## 3. 上传模型权重（Release）

权重不进 git 仓库（249MB，超过 GitHub 单文件 100MB 限制），走 Release：

```bash
# 先生成推理版（剥离优化器状态）
python scripts/export_model.py --ckpt checkpoints/sft.pt --out release/unknow-62m.pt
# 749MB -> 249MB
```

然后在 GitHub 网页端：**Releases → Draft a new release**

| 字段 | 建议值 |
|------|--------|
| Tag | `v1.0` |
| Title | `Unknow 62M v1.0` |
| 附件 | `release/unknow-62m.pt` |
| 说明 | 复制 [docs/BENCHMARK.md](BENCHMARK.md) 的最终成绩表 |

也可以用 CLI：

```bash
gh release create v1.0 release/unknow-62m.pt \
  --title "Unknow 62M v1.0" \
  --notes "见 docs/BENCHMARK.md"
```

用户下载后放到 `checkpoints/sft.pt` 即可直接 `chat.bat` 使用。

## 4. 仓库设置建议

| 项 | 建议 |
|----|------|
| Description | 62M 参数、从零训练的中文工具调用 Agent 模型｜单卡 8GB 十小时复现 |
| Topics | `llm` `small-language-model` `tool-calling` `agent` `from-scratch` `pytorch` `chinese` `cli` |
| About → Website | 可留空，或指向 Release |
| Discussions | 建议开启（复现问题多） |

## 5. 可选：想要更小的本地仓库

旧历史占 5.5GB，确认不再需要后可以清理：

```bash
# 删除旧分支引用（历史提交会变成不可达对象）
git branch -D main-legacy
# 清理不可达对象，回收空间（不可恢复，确认后再执行）
git reflog expire --expire=now --all && git gc --prune=now --aggressive
```

不清理也完全没问题——它不会影响推送，只是本地多占 5.5GB。

## 6. 发布前自检清单

- [ ] `git ls-tree -r --name-only HEAD | wc -l` 是 47
- [ ] `git ls-tree -r -l HEAD | awk '{s+=$4} END {print s/1048576" MB"}'` 约 14.5 MB
- [ ] 没有 `.pt` / `.venv` / `_corpus_cache` 被跟踪：`git ls-files | grep -E "\.pt$|\.venv|_corpus"` 应为空
- [ ] README 里的 Release 链接与实际上传的文件名一致（`unknow-62m.pt`）
- [ ] 语料授权说明在 README 与 LICENSE 都已写明（上游 CC-BY-NC，本仓库不再分发）
- [ ] 在一台干净机器上 `pip install -r requirements.txt && bash scripts/quickstart.sh --smoke` 能跑通
