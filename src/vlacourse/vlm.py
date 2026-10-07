"""Reference vision-language model (the tiny LLaVA-style VLM of Lab 08).

image (B, 3, H, W) --ViT--> patch features (B, N_p, d_v) --projector--> image tokens (B, N_p, d)
[image tokens][text tokens] --TinyGPT--> next-token logits for the text positions.
"""

import torch
import torch.nn as nn

from .transformer import TinyGPT
from .vision import ViT


class TinyVLM(nn.Module):
    def __init__(self, vision: ViT, lm: TinyGPT, d_vision: int, prefix_lm: bool = False, pos2d: bool = False):
        super().__init__()
        self.vision, self.lm, self.prefix_lm = vision, lm, prefix_lm
        d = lm.tok_emb.embedding_dim
        self.projector = nn.Sequential(nn.Linear(d_vision, d), nn.GELU(), nn.Linear(d, d))  # LLaVA-1.5 style
        # Optional factorised 2-D position (row + column) for the image tokens: the LM's own 1-D positions
        # only see the raster order, in which "same column" is a periodic pattern (book 2.3, 3.2).
        self.row_emb = nn.Embedding(vision.grid, d) if pos2d else None
        self.col_emb = nn.Embedding(vision.grid, d) if pos2d else None

    def image_tokens(self, images):
        feats = self.vision(images)                       # (B, [1 +] N_p, d_v)
        if self.vision.cls is not None:
            feats = feats[:, 1:]                          # keep the patch tokens: they say *where*
        tok = self.projector(feats)                       # (B, N_p, d)
        if self.row_emb is not None:
            idx = torch.arange(self.vision.grid, device=tok.device)
            pos = self.row_emb(idx)[:, None, :] + self.col_emb(idx)[None, :, :]   # (g, g, d)
            tok = tok + pos.reshape(1, -1, tok.shape[-1])                        # raster order, like the patches
        return tok

    def forward(self, images, ids):
        """ids: (B, T). Returns logits (B, T, V): row t predicts token t + 1 of the text."""
        img = self.image_tokens(images)
        return self.lm(ids, prefix=img, prefix_lm=self.prefix_lm)[:, img.shape[1]:]

    @torch.no_grad()
    def generate(self, images, ids, n_new: int, eos: int | None = None):
        """Greedy decoding with a KV cache: one prefill over [image; prompt], then one token per step."""
        caches = [{} for _ in self.lm.blocks]
        img = self.image_tokens(images)
        logits = self.lm(ids, caches, prefix=img, prefix_lm=self.prefix_lm)
        pos, out = img.shape[1] + ids.shape[1], []
        for _ in range(n_new):
            nxt = logits[:, -1].argmax(-1, keepdim=True)  # (B, 1)
            out.append(nxt)
            if eos is not None and bool((torch.cat(out, 1) == eos).any(1).all()):
                break
            logits = self.lm(nxt, caches, pos_offset=pos)
            pos += 1
        return torch.cat(out, 1)
