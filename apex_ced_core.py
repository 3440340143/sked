"""
Apex-CED-1.58 -- reference implementation (research prototype).

Implements the architecture described in README.md:

  1. 1.58-bit ternary weights (BitLinear) with a corrected straight-through estimator
  2. Causal encoder-decoder (CED): a shallow causal encoder produces ONE shared
     global KV projection, reused by every decoder layer via cross-attention
  3. Sliding-window local attention in both stacks, so local KV stays O(window)
  4. Causal masking on the cross-attention path (the part that is easy to get wrong)

STATUS: untrained research prototype. The code runs; the attention masks are
covered by tests in `smoke_test.py`. No training run has been performed and no
number in README.md is backed by a measurement.
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1.58-bit ternary linear layer
# ---------------------------------------------------------------------------
class BitLinear158(nn.Module):
    """Ternary weight matmul with 8-bit dynamic activation quantization.

    Forward is exactly the quantized computation; gradients pass through the
    quantizer via a straight-through estimator with the |w| <= 1.5 cut used in
    the BitNet b1.58 recipe.

    The forward value must be the quantized weight itself. Writing
    ``w + (w_q - w).detach()`` (a common shortcut) leaves a residual float term
    in the forward pass and is NOT ternary; the form below avoids that.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False, eps: float = 1e-5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1.0 / math.sqrt(self.in_features) if self.in_features > 0 else 0.0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight
        gamma = w.abs().mean().clamp(min=self.eps)
        w_scaled = w / gamma
        w_quant = torch.clamp(torch.round(w_scaled), -1.0, 1.0)
        mask = (w_scaled.abs() <= 1.5).to(w_scaled.dtype)
        # forward == w_quant exactly; d/dw_scaled == mask
        w_ste = w_quant.detach() + mask * (w_scaled - w_scaled.detach())

        beta = x.abs().amax(dim=-1, keepdim=True).clamp(min=self.eps)
        x_scaled = x * (127.0 / beta)
        x_quant = torch.clamp(torch.round(x_scaled), -128.0, 127.0)
        x_ste = x_quant.detach() + (x_scaled - x_scaled.detach())

        out = F.linear(x_ste, w_ste, self.bias)
        return out * (beta * gamma / 127.0)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(var + self.eps) * self.weight


class RotaryEmbedding(nn.Module):
    """RoPE with explicit absolute positions (so it composes with a KV cache)."""

    def __init__(self, head_dim: int, theta: float = 10000.0):
        super().__init__()
        assert head_dim % 2 == 0, "head_dim must be even for RoPE"
        self.head_dim = head_dim
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor, position_ids: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        # q, k: [B, H, L, D]   position_ids: [B, L]
        freqs = position_ids.float().unsqueeze(-1) * self.inv_freq        # [B, L, D/2]
        emb = torch.cat((freqs, freqs), dim=-1).unsqueeze(1)              # [B, 1, L, D]
        cos, sin = emb.cos(), emb.sin()
        return q * cos + self._rotate_half(q) * sin, k * cos + self._rotate_half(k) * sin


# ---------------------------------------------------------------------------
# Attention helpers
# ---------------------------------------------------------------------------
def sliding_window_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                             window_size: int) -> torch.Tensor:
    """Causal attention restricted to the last `window_size` positions.

    Note: this builds an explicit [Lq, Lk] boolean mask. At 128k context that
    mask alone is O(L^2) memory, so this path is only usable for short
    sequences. A production kernel would need a fused sliding-window
    implementation (e.g. FlashAttention-2's `window_size` argument).
    """
    lq, lk = q.size(2), k.size(2)
    q_idx = torch.arange(lk - lq, lk, device=q.device).unsqueeze(1)   # [Lq, 1]
    k_idx = torch.arange(lk, device=q.device).unsqueeze(0)            # [1, Lk]
    mask = (k_idx <= q_idx) & (k_idx >= q_idx - window_size + 1)      # True = attend
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)


class CausalCEDCrossAttention(nn.Module):
    """Decoder -> shared global KV, with an explicit causal mask.

    The encoder is itself causal, so global_k[j] already summarises tokens
    [0..j]. Without masking j > i the decoder can read the answer it is trying
    to predict. The mask is written in absolute-index form so that it stays
    correct once the cache is populated (q_len == 1 during decode).
    """

    def __init__(self, dim: int, num_heads: int, head_dim: int, cap_val: float = 50.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.cap_val = cap_val
        self.scale = 1.0 / math.sqrt(head_dim)
        self.q_proj = BitLinear158(dim, num_heads * head_dim)
        self.o_proj = BitLinear158(num_heads * head_dim, dim)

    def forward(self, x: torch.Tensor, global_k: torch.Tensor, global_v: torch.Tensor) -> torch.Tensor:
        b, lq, _ = x.shape
        lk = global_k.size(2)
        q = self.q_proj(x).view(b, lq, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, global_k.transpose(-2, -1)) * self.scale
        scores = self.cap_val * torch.tanh(scores / self.cap_val)

        q_idx = torch.arange(lk - lq, lk, device=x.device).unsqueeze(1)
        k_idx = torch.arange(lk, device=x.device).unsqueeze(0)
        future = (k_idx > q_idx).unsqueeze(0).unsqueeze(1)
        scores = scores.masked_fill(future, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        ctx = torch.matmul(attn, global_v).transpose(1, 2).reshape(b, lq, -1)
        return self.o_proj(ctx)


# ---------------------------------------------------------------------------
# Feed-forward stacks
# ---------------------------------------------------------------------------
class DenseSwiGLU(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.gate = BitLinear158(dim, hidden)
        self.up = BitLinear158(dim, hidden)
        self.down = BitLinear158(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class TernaryMoE(nn.Module):
    """Sigmoid-gated ternary expert pool with a loss-free balancing bias.

    Tokens are dispatched by grouping on expert id (one pass per expert, not
    one pass per (token, expert) pair) and accumulated with index_add_.
    """

    def __init__(self, dim: int, hidden: int, num_experts: int = 32, top_k: int = 4,
                 bias_lr: float = 1e-3):
        super().__init__()
        assert 0 < top_k <= num_experts
        self.num_experts = num_experts
        self.top_k = top_k
        self.bias_lr = bias_lr
        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.register_buffer("expert_bias", torch.zeros(num_experts))
        self.up = nn.ModuleList([BitLinear158(dim, hidden) for _ in range(num_experts)])
        self.gate_proj = nn.ModuleList([BitLinear158(dim, hidden) for _ in range(num_experts)])
        self.down = nn.ModuleList([BitLinear158(hidden, dim) for _ in range(num_experts)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, l, d = x.shape
        x_flat = x.reshape(-1, d)
        n_tokens = x_flat.size(0)

        scores = torch.sigmoid(self.gate(x_flat)) + self.expert_bias
        topk_scores, topk_idx = torch.topk(scores, self.top_k, dim=-1)
        topk_w = F.softmax(topk_scores, dim=-1)

        if self.training:
            with torch.no_grad():
                counts = torch.bincount(topk_idx.reshape(-1), minlength=self.num_experts)
                counts = counts.to(self.expert_bias.dtype)
                target = (n_tokens * self.top_k) / self.num_experts
                self.expert_bias.add_(self.bias_lr * (target - counts).sign())

        out = torch.zeros_like(x_flat)
        flat_expert = topk_idx.reshape(-1)
        flat_token = torch.arange(n_tokens, device=x.device).repeat_interleave(self.top_k)
        flat_w = topk_w.reshape(-1, 1)

        for e in range(self.num_experts):
            sel = (flat_expert == e).nonzero(as_tuple=True)[0]
            if sel.numel() == 0:
                continue
            tok = flat_token[sel]
            h = F.silu(self.gate_proj[e](x_flat[tok])) * self.up[e](x_flat[tok])
            out.index_add_(0, tok, self.down[e](h) * flat_w[sel])

        return out.reshape(b, l, d)


# ---------------------------------------------------------------------------
# Blocks
# ---------------------------------------------------------------------------
@dataclass
class KVCache:
    k: Optional[torch.Tensor] = None
    v: Optional[torch.Tensor] = None


class ApexCEDBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, is_decoder: bool, window_size: int,
                 ffn_hidden: int, num_experts: int = 32, top_k: int = 4):
        super().__init__()
        self.is_decoder = is_decoder
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.norm1 = RMSNorm(dim)
        self.q_proj = BitLinear158(dim, dim)
        self.k_proj = BitLinear158(dim, dim)
        self.v_proj = BitLinear158(dim, dim)
        self.o_proj = BitLinear158(dim, dim)

        self.norm2 = RMSNorm(dim)
        if is_decoder:
            self.norm_cross = RMSNorm(dim)
            self.cross_attn = CausalCEDCrossAttention(dim, num_heads, self.head_dim)
            self.ffn = TernaryMoE(dim, ffn_hidden, num_experts, top_k)
        else:
            self.ffn = DenseSwiGLU(dim, ffn_hidden)

    def forward(self, x: torch.Tensor, rope: RotaryEmbedding, position_ids: torch.Tensor,
                cache: Optional[KVCache] = None,
                global_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> torch.Tensor:
        b, l, d = x.shape
        h = self.norm1(x)
        q = self.q_proj(h).view(b, l, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(b, l, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(b, l, self.num_heads, self.head_dim).transpose(1, 2)
        q, k = rope(q, k, position_ids)

        if cache is not None:
            if cache.k is not None:
                k = torch.cat([cache.k, k], dim=2)
                v = torch.cat([cache.v, v], dim=2)
            # attend over the full (untruncated) history; the window mask decides
            # what is visible, while the stored cache stays O(window)
            cache.k = k[:, :, -self.window_size:, :].detach()
            cache.v = v[:, :, -self.window_size:, :].detach()

        attn = sliding_window_attention(q, k, v, self.window_size)
        x = x + self.o_proj(attn.transpose(1, 2).reshape(b, l, d))

        if self.is_decoder and global_kv is not None:
            x = x + self.cross_attn(self.norm_cross(x), global_kv[0], global_kv[1])
        x = x + self.ffn(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class ApexCEDModel(nn.Module):
    def __init__(self, vocab_size: int = 128256, dim: int = 2048, num_heads: int = 16,
                 enc_layers: int = 16, dec_layers: int = 16, enc_ffn: int = 5632,
                 moe_hidden: int = 2816, num_experts: int = 32, top_k: int = 4,
                 window_size: int = 512, logit_cap: float = 30.0):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.logit_cap = logit_cap
        self.window_size = window_size

        self.embed = nn.Embedding(vocab_size, dim)
        self.rope = RotaryEmbedding(self.head_dim)

        self.encoders = nn.ModuleList([
            ApexCEDBlock(dim, num_heads, False, window_size, enc_ffn)
            for _ in range(enc_layers)
        ])
        self.global_k_proj = BitLinear158(dim, dim)
        self.global_v_proj = BitLinear158(dim, dim)

        self.decoders = nn.ModuleList([
            ApexCEDBlock(dim, num_heads, True, window_size, moe_hidden, num_experts, top_k)
            for _ in range(dec_layers)
        ])
        self.final_norm = RMSNorm(dim)
        # weight-tied output head; the bias vector is kept for parity with the spec
        self.lm_head_bias = nn.Parameter(torch.zeros(vocab_size))

    # -- core step, shared by training forward and incremental decode ----------
    def _step(self, input_ids: torch.Tensor, position_ids: torch.Tensor,
              enc_caches: Optional[List[KVCache]], dec_caches: Optional[List[KVCache]],
              global_kv: Optional[Tuple[torch.Tensor, torch.Tensor]]
              ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        b, l = input_ids.shape
        h = self.embed(input_ids)

        for i, blk in enumerate(self.encoders):
            h = blk(h, self.rope, position_ids,
                    cache=None if enc_caches is None else enc_caches[i])

        nk = self.global_k_proj(h).view(b, l, self.num_heads, self.head_dim).transpose(1, 2)
        nv = self.global_v_proj(h).view(b, l, self.num_heads, self.head_dim).transpose(1, 2)
        if global_kv is None:
            gk, gv = nk, nv
        else:
            gk = torch.cat([global_kv[0], nk], dim=2)
            gv = torch.cat([global_kv[1], nv], dim=2)

        d = h
        for i, blk in enumerate(self.decoders):
            d = blk(d, self.rope, position_ids,
                    cache=None if dec_caches is None else dec_caches[i],
                    global_kv=(gk, gv))

        logits = F.linear(self.final_norm(d), self.embed.weight, self.lm_head_bias)
        return self.logit_cap * torch.tanh(logits / self.logit_cap), (gk, gv)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        b, l = input_ids.shape
        pos = torch.arange(l, device=input_ids.device).unsqueeze(0).expand(b, l)
        logits, _ = self._step(input_ids, pos, None, None, None)
        return logits

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 16,
                 temperature: float = 0.0) -> torch.Tensor:
        """Greedy / sampled incremental decode.

        Encoder and decoder self-attention caches stay at O(window); the shared
        global KV grows linearly with the sequence (one copy for the whole
        decoder, rather than one per layer).
        """
        was_training = self.training
        self.eval()
        b, l = input_ids.shape
        enc_caches = [KVCache() for _ in self.encoders]
        dec_caches = [KVCache() for _ in self.decoders]

        pos = torch.arange(l, device=input_ids.device).unsqueeze(0).expand(b, l)
        logits, global_kv = self._step(input_ids, pos, enc_caches, dec_caches, None)
        out = input_ids

        for step in range(max_new_tokens):
            nxt = logits[:, -1, :]
            if temperature <= 0:
                nxt = nxt.argmax(dim=-1, keepdim=True)
            else:
                nxt = torch.multinomial(F.softmax(nxt / temperature, dim=-1), 1)
            out = torch.cat([out, nxt], dim=1)
            pos = torch.full((b, 1), l + step, device=input_ids.device, dtype=torch.long)
            logits, global_kv = self._step(nxt, pos, enc_caches, dec_caches, global_kv)

        if was_training:
            self.train()
        return out


# ---------------------------------------------------------------------------
# Accounting helpers (used by smoke_test.py / README)
# ---------------------------------------------------------------------------
def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def count_parameters_by_kind(model: nn.Module) -> dict:
    ternary = sum(p.numel() for m in model.modules() if isinstance(m, BitLinear158)
                  for p in [m.weight])
    return {"ternary_weights": ternary, "total": count_parameters(model)}


def estimate_static_bytes(model: nn.Module, ternary_bits: float = 2.0) -> dict:
    """Analytical static weight footprint.

    Ternary weights are packed at `ternary_bits` per parameter (1.58 rounds up to
    2 bits of storage); every other parameter is counted at its own dtype width.
    This is arithmetic, not a measurement.
    """
    ternary_ids = {id(m.weight) for m in model.modules() if isinstance(m, BitLinear158)}
    ternary_bytes = other_bytes = 0
    for p in model.parameters():
        if id(p) in ternary_ids:
            ternary_bytes += p.numel() * ternary_bits / 8.0
        else:
            other_bytes += p.numel() * p.element_size()
    return {
        "ternary_bytes": ternary_bytes,
        "other_bytes": other_bytes,
        "total_bytes": ternary_bytes + other_bytes,
    }