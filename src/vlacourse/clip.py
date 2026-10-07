"""Reference contrastive image-text model (the tiny CLIP of Lab 07) and its two losses."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import causal_mask
from .transformer import Block, RMSNorm
from .vision import ViT


class TextEncoder(nn.Module):
    """CLIP-style text tower: causal transformer, the output at the <eos> position is the sentence vector."""

    def __init__(self, vocab_size: int, d: int = 128, n_layers: int = 3, n_heads: int = 4, max_len: int = 32):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d)
        nn.init.normal_(self.tok.weight, std=0.02)
        self.pos = nn.Parameter(torch.randn(1, max_len, d) * 0.02)
        self.blocks = nn.ModuleList(Block(d, n_heads, 4 * d) for _ in range(n_layers))
        self.norm = RMSNorm(d)

    def forward(self, ids, eos_pos):
        """ids: (B, T); eos_pos: (B,) index of <eos> in each row. Returns (B, d)."""
        T = ids.shape[1]
        x = self.tok(ids) + self.pos[:, :T]
        mask = causal_mask(T, ids.device)            # padding after <eos> cannot leak into it
        for blk in self.blocks:
            x = blk(x, mask=mask)
        return self.norm(x)[torch.arange(len(ids)), eos_pos]


class TinyCLIP(nn.Module):
    """Two towers + linear projections into a shared d_e-dimensional, L2-normalised space."""

    def __init__(self, vocab_size: int, img_size: int = 64, patch: int = 8, d: int = 128, d_e: int = 64,
                 init_temperature: float = 0.07):
        super().__init__()
        self.image_encoder = ViT(img_size, patch, d, n_layers=4, n_heads=4)
        self.text_encoder = TextEncoder(vocab_size, d)
        self.image_proj = nn.Linear(d, d_e, bias=False)
        self.text_proj = nn.Linear(d, d_e, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / init_temperature)))  # learned 1/temperature
        self.logit_bias = nn.Parameter(torch.tensor(-10.0))                              # used by the SigLIP loss

    def encode_image(self, images):
        return F.normalize(self.image_proj(self.image_encoder.pooled(images)), dim=-1)   # (B, d_e)

    def encode_text(self, ids, eos_pos):
        return F.normalize(self.text_proj(self.text_encoder(ids, eos_pos)), dim=-1)     # (B, d_e)


def clip_loss(img_emb, txt_emb, logit_scale):
    """Symmetric InfoNCE (CLIP). img_emb, txt_emb: (B, d_e), L2-normalised; pair i matches pair i.

    Row i of `logits` is a B-way classification "which caption belongs to image i?"; the
    columns answer the transposed question. Both use the diagonal as the correct class.
    """
    logits = logit_scale.exp().clamp(max=100) * img_emb @ txt_emb.T            # (B, B)
    labels = torch.arange(len(logits), device=logits.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2


def siglip_loss(img_emb, txt_emb, logit_scale, logit_bias):
    """Sigmoid loss (SigLIP): an independent binary 'match / no match' decision for every one of the B^2 pairs."""
    logits = logit_scale.exp() * img_emb @ txt_emb.T + logit_bias            # (B, B)
    z = 2 * torch.eye(len(logits), device=logits.device) - 1                  # +1 on the diagonal, -1 elsewhere
    return -F.logsigmoid(z * logits).sum() / len(logits)
