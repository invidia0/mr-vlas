"""Reference attention implementation (the one built by hand in Lab 03).

Shape conventions (see the book's notation appendix):
    B batch, T query length, S key/value length, d model width,
    h heads, d_h = d / h per-head width.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_attention(q, k, v, mask=None):
    """softmax(q k^T / sqrt(d_h)) v.

    q: (..., T, d_h), k: (..., S, d_h), v: (..., S, d_v)
    mask: bool, broadcastable to (..., T, S); True = may attend.
    Returns (output (..., T, d_v), weights (..., T, S)).
    """
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])  # (..., T, S)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    weights = scores.softmax(dim=-1)  # each row sums to 1 over the S keys
    return weights @ v, weights


def causal_mask(T: int, device=None) -> torch.Tensor:
    """(T, T) lower-triangular bool mask: position i may attend to j <= i."""
    return torch.ones(T, T, dtype=torch.bool, device=device).tril()


class MultiHeadAttention(nn.Module):
    """Multi-head attention; self-attention if `context` is None, else cross-attention."""

    def __init__(self, d_model: int, n_heads: int, bias: bool = False):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.h, self.d_h = n_heads, d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.o_proj = nn.Linear(d_model, d_model, bias=bias)
        self.last_weights = None  # (B, h, T, S), kept for inspection in the labs

    def _split(self, x):  # (B, T, d) -> (B, h, T, d_h)
        B, T, _ = x.shape
        return x.view(B, T, self.h, self.d_h).transpose(1, 2)

    def forward(self, x, context=None, mask=None, kv_cache=None):
        """x: (B, T, d). context: (B, S, d) or None. mask: broadcastable to (B, h, T, S).

        kv_cache: optional dict; if given, new keys/values are appended to
        kv_cache["k"], kv_cache["v"] (shape (B, h, S_past, d_h)) and the full
        cache is attended over. Used for incremental decoding.
        """
        src = x if context is None else context
        q, k, v = self._split(self.q_proj(x)), self._split(self.k_proj(src)), self._split(self.v_proj(src))
        if kv_cache is not None:
            if "k" in kv_cache:
                k = torch.cat([kv_cache["k"], k], dim=2)
                v = torch.cat([kv_cache["v"], v], dim=2)
            kv_cache["k"], kv_cache["v"] = k, v
        out, w = scaled_dot_product_attention(q, k, v, mask)  # (B, h, T, d_h)
        self.last_weights = w.detach()
        B, _, T, _ = out.shape
        return self.o_proj(out.transpose(1, 2).reshape(B, T, self.h * self.d_h))
