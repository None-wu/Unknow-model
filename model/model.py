"""40M 级微型 GPT：纯 PyTorch 实现，8GB 显存可训。

结构选择理由（都是为了在 4060 Laptop 上跑得动又不至于太笨）：
- GQA(kv_heads=2)：KV cache 只有 MHA 的 1/4，推理时上下文能开更长；
- SwiGLU + RMSNorm + RoPE + 权重绑定：同参数量下效果最好的常规组合；
- 无 bias、无 dropout：小模型上 dropout 反而拖慢收敛，数据量足够；
- 词表 16384：中文 + 命令语法共用，比 32k 省 2.6M 参数给到 FFN。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 16384
    dim: int = 576
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: int = 2
    ffn_dim: int = 1536
    max_seq_len: int = 1024
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    tie_embeddings: bool = True

    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads

    def num_params(self) -> int:
        emb = self.vocab_size * self.dim
        per_layer = (
            self.dim * self.dim                      # q
            + 2 * self.dim * (self.n_kv_heads * self.head_dim)  # k, v
            + self.dim * self.dim                    # o
            + 3 * self.dim * self.ffn_dim            # gate, up, down
        )
        total = emb + per_layer * self.n_layers + self.n_layers * 2 * self.dim
        return total if self.tie_embeddings else total + emb


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


def build_rope(head_dim: int, max_seq: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_seq).float()
    freqs = torch.outer(t, inv)          # (seq, head_dim/2)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (B, H, T, D)  cos/sin: (T, D)
    return x * cos[None, None, : x.shape[2], :] + rotate_half(x) * sin[None, None, : x.shape[2], :]


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.q = nn.Linear(cfg.dim, cfg.n_heads * cfg.head_dim, bias=False)
        self.k = nn.Linear(cfg.dim, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.v = nn.Linear(cfg.dim, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.o = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.dim, bias=False)

    def forward(self, x, cos, sin, kv_cache=None, start_pos: int = 0):
        B, T, _ = x.shape
        h, kvh, d = self.cfg.n_heads, self.cfg.n_kv_heads, self.cfg.head_dim
        q = self.q(x).view(B, T, h, d).transpose(1, 2)
        k = self.k(x).view(B, T, kvh, d).transpose(1, 2)
        v = self.v(x).view(B, T, kvh, d).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=2)
            v = torch.cat([kv_cache[1], v], dim=2)
        new_cache = (k, v)

        k = k.repeat_interleave(h // kvh, dim=1)
        v = v.repeat_interleave(h // kvh, dim=1)

        # 训练时整段（无 cache）走 is_causal；推理单步不加 mask
        causal = T > 1 and kv_cache is None
        out = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        out = out.transpose(1, 2).contiguous().view(B, T, h * d)
        return self.o(out), new_cache


class FeedForward(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate = nn.Linear(cfg.dim, cfg.ffn_dim, bias=False)
        self.up = nn.Linear(cfg.dim, cfg.ffn_dim, bias=False)
        self.down = nn.Linear(cfg.ffn_dim, cfg.dim, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n1 = RMSNorm(cfg.dim, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.n2 = RMSNorm(cfg.dim, cfg.norm_eps)
        self.ffn = FeedForward(cfg)

    def forward(self, x, cos, sin, kv_cache=None):
        h, cache = self.attn(self.n1(x), cos, sin, kv_cache)
        x = x + h
        x = x + self.ffn(self.n2(x))
        return x, cache


class MiniGPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight

        cos, sin = build_rope(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.apply(self._init)
        # 残差分支缩放初始化，深度堆叠时更稳
        for name, p in self.named_parameters():
            if name.endswith("o.weight") or name.endswith("down.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layers))

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None,
                loss_mask: torch.Tensor | None = None):
        B, T = idx.shape
        x = self.embed(idx)
        cos = self.cos[:T]
        sin = self.sin[:T]
        for blk in self.blocks:
            x, _ = blk(x, cos, sin)
        x = self.norm(x)
        logits = self.lm_head(x)

        if targets is None:
            return logits, None

        if loss_mask is None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)).float(),
                targets.reshape(-1),
                ignore_index=-100,
            )
        else:
            flat_logits = logits.view(-1, logits.size(-1)).float()
            flat_t = targets.reshape(-1)
            flat_m = loss_mask.reshape(-1)
            sel = flat_m > 0
            if sel.sum() == 0:
                loss = flat_logits.sum() * 0.0
            else:
                loss = F.cross_entropy(flat_logits[sel], flat_t[sel])
        return logits, loss

    @torch.inference_mode()
    def generate(self, idx: torch.Tensor, max_new_tokens: int = 128,
                 temperature: float = 0.8, top_k: int | None = 40,
                 stop_ids: set[int] | None = None,
                 logits_filter=None, on_token=None, repeat_stop: int = 16):
        """自回归生成。

        logits_filter(step, logits) -> logits  受约束解码
        on_token(token_id)                     每采样一步回调，用于推进语法状态
        """
        self.eval()
        caches = [None] * self.cfg.n_layers
        out = idx
        gen = idx
        run_len, last_tok = 0, -1
        for step in range(max_new_tokens):
            cur = gen[:, -1:] if step > 0 else gen
            T_cur = cur.shape[1]
            start = gen.shape[1] - T_cur
            x = self.embed(cur)
            cos = self.cos[start:start + T_cur]
            sin = self.sin[start:start + T_cur]
            new_caches = []
            for i, blk in enumerate(self.blocks):
                x, c = blk(x, cos, sin, caches[i])
                new_caches.append(c)
            caches = new_caches
            logits = self.lm_head(self.norm(x))[:, -1, :].float()

            if logits_filter is not None:
                logits = logits_filter(step, logits)

            if temperature <= 0:
                nxt = logits.argmax(-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k:
                    k = min(top_k, logits.size(-1))
                    thresh = torch.topk(logits, k)[0][..., -1, None]
                    logits = logits.masked_fill(logits < thresh, float("-inf"))
                probs = F.softmax(logits, dim=-1)
                nxt = torch.multinomial(probs, 1)
            gen = torch.cat([gen, nxt], dim=1)
            if on_token is not None:
                on_token(int(nxt.item()))
            tid = int(nxt.item())
            run_len = run_len + 1 if tid == last_tok else 1
            last_tok = tid
            if stop_ids and tid in stop_ids:
                break
            # 小模型未训好时会陷入复读，超过 repeat_stop 直接截断
            if repeat_stop and run_len >= repeat_stop:
                break
        return gen[:, idx.shape[1]:]


if __name__ == "__main__":
    cfg = ModelConfig()
    m = MiniGPT(cfg)
    n = m.num_params()
    print(f"参数总量: {n/1e6:.1f}M")
    print(f"  其中 embedding: {cfg.vocab_size*cfg.dim/1e6:.1f}M（与 lm_head 共享）")
    print(f"  每层: {(n - cfg.vocab_size*cfg.dim)/cfg.n_layers/1e6:.2f}M")
    x = torch.randint(0, cfg.vocab_size, (2, 64))
    logits, _ = m(x)
    print("前向输出:", tuple(logits.shape))
    loss = m(x, x)[1]
    print(f"随机初始化 loss: {loss.item():.3f}（理论 ln(16384)={math.log(cfg.vocab_size):.3f}）")
    out = m.generate(x[:1, :8], max_new_tokens=5, temperature=0)
    print("生成:", out.shape)
