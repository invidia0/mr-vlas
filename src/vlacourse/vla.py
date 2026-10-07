"""A toy Vision-Language-Action model (Lab 11; book Chapter 8).

Observation tokens: 64 image-patch tokens (ViT), the instruction's word tokens, one proprioception token.
A bidirectional transformer backbone processes them (the "VLM"), and one of three action heads produces an
action chunk A_t of shape (H_a, d_a):

* "mse"    — regression from the pooled backbone output (the baseline that averages modes, book 6.1);
* "tokens" — RT-2/OpenVLA style: each action dimension of each step is a token from n_bins bins, generated
             autoregressively *by the backbone itself* (prefix-LM mask), so every token costs a backbone pass;
* "flow"   — π0/GR00T style: a small action expert denoises the whole chunk by flow matching, cross-attending
             to the backbone's output tokens, which are computed once per chunk.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import MultiHeadAttention
from .generative import Binner
from .transformer import Block, RMSNorm, SwiGLU
from .vision import ViT


def time_features(tau, dim):
    """Sinusoidal features of a flow time tau in [0, 1]; (B,) -> (B, dim)."""
    f = torch.exp(-math.log(1000.0) * torch.arange(dim // 2, device=tau.device) / (dim // 2))
    a = tau[:, None] * 1000.0 * f
    return torch.cat([a.sin(), a.cos()], -1)


class CrossBlock(nn.Module):
    """x <- x + SelfAttn(Norm x); x <- x + CrossAttn(Norm x, context); x <- x + FFN(Norm x)."""

    def __init__(self, d, n_heads):
        super().__init__()
        self.n1, self.self_attn = RMSNorm(d), MultiHeadAttention(d, n_heads)
        self.n2, self.cross_attn = RMSNorm(d), MultiHeadAttention(d, n_heads)
        self.n3, self.ffn = RMSNorm(d), SwiGLU(d, 4 * d)

    def forward(self, x, context, context_mask=None):
        x = x + self.self_attn(self.n1(x))
        x = x + self.cross_attn(self.n2(x), context=context, mask=context_mask)
        return x + self.ffn(self.n3(x))


class FlowExpert(nn.Module):
    """Velocity network v(A_tau, tau | context) over the H_a actions of a chunk."""

    def __init__(self, d, d_action, horizon, n_layers=2, n_heads=4):
        super().__init__()
        self.inp = nn.Linear(d_action, d)
        self.pos = nn.Parameter(torch.randn(1, horizon, d) * 0.02)
        self.time = nn.Sequential(nn.Linear(64, d), nn.SiLU(), nn.Linear(d, d))
        self.blocks = nn.ModuleList(CrossBlock(d, n_heads) for _ in range(n_layers))
        self.out = nn.Sequential(RMSNorm(d), nn.Linear(d, d_action))

    def forward(self, a_tau, tau, context, context_mask=None):          # a_tau: (B, H_a, d_a)
        x = self.inp(a_tau) + self.pos + self.time(time_features(tau, 64))[:, None, :]
        for blk in self.blocks:
            x = blk(x, context, context_mask)
        return self.out(x)                                                # (B, H_a, d_a) velocity


class TinyVLA(nn.Module):
    def __init__(self, vocab_size, head="flow", d=128, n_layers=4, n_heads=4, img_size=64, patch=8,
                 d_state=2, d_action=2, horizon=8, n_bins=32, action_low=-1.0, action_high=1.0,
                 vision: ViT | None = None, d_vision=128):
        super().__init__()
        assert head in ("mse", "tokens", "flow")
        self.head, self.horizon, self.d_action, self.n_bins = head, horizon, d_action, n_bins
        self.vision = vision or ViT(img_size, patch, d_vision, n_layers=4, n_heads=n_heads)
        self.n_img = self.vision.grid ** 2
        self.img_proj = nn.Linear(d_vision, d)
        self.word = nn.Embedding(vocab_size, d)
        nn.init.normal_(self.word.weight, std=0.02)
        self.state_proj = nn.Linear(d_state, d)
        self.segment = nn.Embedding(4, d)              # 0 image, 1 text, 2 state, 3 action token
        self.pos = nn.Parameter(torch.randn(1, 128, d) * 0.02)
        self.blocks = nn.ModuleList(Block(d, n_heads, 4 * d) for _ in range(n_layers))
        self.norm = RMSNorm(d)
        self.binner = Binner([action_low] * d_action, [action_high] * d_action, K=n_bins)
        if head == "mse":
            self.mse_head = nn.Sequential(nn.Linear(d, 4 * d), nn.SiLU(), nn.Linear(4 * d, horizon * d_action))
        elif head == "tokens":
            self.action_emb = nn.Embedding(n_bins + 1, d)  # + a start token
            self.bin_head = nn.Linear(d, n_bins)
        else:
            self.expert = FlowExpert(d, d_action, horizon, n_heads=n_heads)

    # ------------------------------------------------------------------ backbone
    def _prefix(self, images, instr, state):
        """Observation tokens (B, P, d) and key mask (B, P): True = real token."""
        img = self.img_proj(self.vision(images)[:, 1:])                       # (B, 64, d) patch tokens
        txt = self.word(instr)                                                # (B, L, d)
        st = self.state_proj(state)[:, None, :]                               # (B, 1, d)
        seg = torch.cat([torch.zeros(img.shape[1], dtype=torch.long), torch.ones(txt.shape[1], dtype=torch.long),
                         torch.full((1,), 2)]).to(images.device)
        x = torch.cat([img, txt, st], 1) + self.segment(seg)
        mask = torch.cat([torch.ones(img.shape[:2], dtype=torch.bool, device=images.device), instr != 0,
                          torch.ones(st.shape[:2], dtype=torch.bool, device=images.device)], 1)
        return x, mask

    def _run(self, x, attn_mask):
        x = x + self.pos[:, :x.shape[1]]
        for blk in self.blocks:
            x = blk(x, mask=attn_mask)
        return self.norm(x)

    def encode(self, images, instr, state):
        """One backbone pass over the observation; returns context (B, P, d) and its key mask (B, P)."""
        x, key = self._prefix(images, instr, state)
        return self._run(x, key[:, None, None, :]), key

    # ------------------------------------------------------------------ token head (backbone decodes)
    def _token_logits(self, images, instr, state, ids):
        """ids: (B, n) action-token inputs (start token first). Prefix-LM mask: the observation attends to
        itself bidirectionally and never to action tokens; action token i sees the observation and tokens <= i."""
        x, key = self._prefix(images, instr, state)
        P, n = x.shape[1], ids.shape[1]
        a = self.action_emb(ids) + self.segment.weight[3]
        T = P + n
        m = torch.zeros(len(x), T, T, dtype=torch.bool, device=x.device)
        m[:, :, :P] = key[:, None, :]                                          # everyone reads real prefix tokens
        m[:, P:, P:] = torch.ones(n, n, dtype=torch.bool, device=x.device).tril()
        m[:, :P, P:] = False                                                   # the prefix never reads actions
        h = self._run(torch.cat([x, a], 1), m[:, None])
        return self.bin_head(h[:, P:])                                         # (B, n, n_bins)

    # ------------------------------------------------------------------ training
    def loss(self, images, instr, state, chunk):
        """chunk: (B, H_a, d_a) target actions in [-1, 1]."""
        B = len(chunk)
        if self.head == "mse":
            ctx, key = self.encode(images, instr, state)
            pooled = (ctx * key[..., None]).sum(1) / key.sum(1, keepdim=True)
            return ((self.mse_head(pooled).view(B, self.horizon, self.d_action) - chunk) ** 2).mean()
        if self.head == "tokens":
            target = self.binner.encode(chunk).flatten(1)                      # (B, H_a * d_a): step-major order
            start = torch.full((B, 1), self.n_bins, device=chunk.device)
            logits = self._token_logits(images, instr, state, torch.cat([start, target[:, :-1]], 1))
            return F.cross_entropy(logits.reshape(-1, self.n_bins), target.reshape(-1))
        ctx, key = self.encode(images, instr, state)
        x0 = torch.randn_like(chunk)
        tau = torch.rand(B, device=chunk.device)
        t = tau[:, None, None]
        v = self.expert((1 - t) * x0 + t * chunk, tau, ctx, key[:, None, None, :])
        return ((v - (chunk - x0)) ** 2).mean()

    # ------------------------------------------------------------------ acting
    @torch.no_grad()
    def act(self, images, instr, state, steps=10, temperature=1.0, generator=None):
        """Sample an action chunk (B, H_a, d_a)."""
        B = len(images)
        if self.head == "mse":
            ctx, key = self.encode(images, instr, state)
            pooled = (ctx * key[..., None]).sum(1) / key.sum(1, keepdim=True)
            return self.mse_head(pooled).view(B, self.horizon, self.d_action)
        if self.head == "tokens":
            ids = torch.full((B, 1), self.n_bins, device=images.device)
            for _ in range(self.horizon * self.d_action):                     # one backbone pass per token
                logits = self._token_logits(images, instr, state, ids)[:, -1] / max(temperature, 1e-6)
                nxt = logits.argmax(-1, keepdim=True) if temperature == 0 else \
                    torch.multinomial(logits.softmax(-1), 1, generator=generator)
                ids = torch.cat([ids, nxt], 1)
            return self.binner.decode(ids[:, 1:].view(B, self.horizon, self.d_action))
        ctx, key = self.encode(images, instr, state)                          # backbone once per chunk
        x = torch.randn(B, self.horizon, self.d_action, device=images.device, generator=generator)
        for i in range(steps):                                                # expert `steps` times
            tau = torch.full((B,), i / steps, device=images.device)
            x = x + self.expert(x, tau, ctx, key[:, None, None, :]) / steps
        return x
