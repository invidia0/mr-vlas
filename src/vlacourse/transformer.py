"""Reference pre-norm decoder-only transformer (the one built in Lab 04)."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import MultiHeadAttention, causal_mask


class RMSNorm(nn.Module):
    """x / rms(x) * g, with rms(x) = sqrt(mean(x^2) + eps) over the last dim."""

    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class SwiGLU(nn.Module):
    """Gated feed-forward: W2 (silu(W1 x) * W3 x)."""

    def __init__(self, d: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d, d_ff, bias=False)
        self.w3 = nn.Linear(d, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    """x <- x + Attn(Norm(x));  x <- x + FFN(Norm(x))."""

    def __init__(self, d: int, n_heads: int, d_ff: int):
        super().__init__()
        self.norm1, self.attn = RMSNorm(d), MultiHeadAttention(d, n_heads)
        self.norm2, self.ffn = RMSNorm(d), SwiGLU(d, d_ff)

    def forward(self, x, mask=None, kv_cache=None):
        x = x + self.attn(self.norm1(x), mask=mask, kv_cache=kv_cache)
        return x + self.ffn(self.norm2(x))


class TinyGPT(nn.Module):
    """Decoder-only language model with learned absolute positions and tied embeddings."""

    def __init__(self, vocab_size: int, d: int = 128, n_layers: int = 4, n_heads: int = 4, max_len: int = 256):
        super().__init__()
        self.max_len = max_len
        self.tok_emb = nn.Embedding(vocab_size, d)
        self.pos_emb = nn.Embedding(max_len, d)
        self.blocks = nn.ModuleList(Block(d, n_heads, 4 * d) for _ in range(n_layers))
        self.norm = RMSNorm(d)
        self.head = nn.Linear(d, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # weight tying
        # nn.Embedding defaults to N(0, 1); with tied weights that makes the initial logits huge
        # (std ~ sqrt(d)) and the initial loss far above log V. GPT-2 uses std 0.02.
        nn.init.normal_(self.tok_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)

    def forward(self, idx, caches=None, pos_offset: int = 0, prefix=None, prefix_lm: bool = False):
        """idx: (B, T) token ids -> logits (B, T, V).

        With caches (a list of dicts, one per layer) the call processes only the
        new tokens idx, which sit at positions pos_offset .. pos_offset+T-1.

        prefix: optional (B, P, d) embeddings placed before the tokens (e.g. projected
        image tokens in a VLM); the logits then cover all P + T positions. With
        prefix_lm=True the prefix attends to itself bidirectionally (PaliGemma-style).
        """
        x = self.tok_emb(idx)  # (B, T, d)
        if prefix is not None:
            x = torch.cat([prefix, x], 1)  # (B, P + T, d)
        T = x.shape[1]
        if pos_offset + T > self.max_len:
            raise ValueError(f"sequence length {pos_offset + T} exceeds max_len={self.max_len} (learned positions)")
        x = x + self.pos_emb(torch.arange(pos_offset, pos_offset + T, device=idx.device))
        # New query i (absolute position pos_offset+i) may see keys 0..pos_offset+i.
        mask = causal_mask(pos_offset + T, idx.device)[pos_offset:]  # (T, pos_offset+T)
        if prefix is not None and prefix_lm:
            if pos_offset:
                raise ValueError("a prefix is only supported in the first (prefill) call")
            P = prefix.shape[1]
            mask = mask.clone()
            mask[:P, :P] = True
        for i, blk in enumerate(self.blocks):
            x = blk(x, mask=mask, kv_cache=None if caches is None else caches[i])
        return self.head(self.norm(x))

    @torch.no_grad()
    def generate(self, idx, n_new: int, temperature: float = 1.0, use_cache: bool = True, generator=None):
        """Sample n_new tokens autoregressively. temperature=0 means greedy."""
        caches = [{} for _ in self.blocks] if use_cache else None
        logits = self(idx, caches)
        for _ in range(n_new):
            last = logits[:, -1]
            if temperature == 0:
                nxt = last.argmax(-1, keepdim=True)
            else:
                nxt = torch.multinomial((last / temperature).softmax(-1), 1, generator=generator)
            if use_cache:
                logits = self(nxt, caches, pos_offset=idx.shape[1])
                idx = torch.cat([idx, nxt], 1)
            else:
                idx = torch.cat([idx, nxt], 1)
                logits = self(idx)
        return idx
