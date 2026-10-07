"""Reference Vision Transformer (the one built in Lab 06).

Shapes: images (B, C, H, W) -> patch tokens (B, N_p, d), N_p = (H / P) * (W / P).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer import Block, RMSNorm


class PatchEmbed(nn.Module):
    """Cut into P x P patches and project each linearly to width d.

    A Conv2d with kernel = stride = P is exactly "flatten each patch, multiply by W_E":
    the kernel is W_E reshaped to (d, C, P, P).
    """

    def __init__(self, patch: int, d: int, channels: int = 3):
        super().__init__()
        self.patch = patch
        self.proj = nn.Conv2d(channels, d, kernel_size=patch, stride=patch)

    def forward(self, x):                                   # (B, C, H, W)
        return self.proj(x).flatten(2).transpose(1, 2)     # (B, N_p, d), patches in raster order


class ViT(nn.Module):
    """Encoder-only transformer over patches, with an optional [CLS] token and learned positions.

    forward returns all output tokens (B, 1 + N_p, d) if cls_token else (B, N_p, d);
    `pooled` gives one vector per image ([CLS] output, or the mean of the patch tokens).
    """

    def __init__(self, img_size: int = 64, patch: int = 8, d: int = 128, n_layers: int = 4, n_heads: int = 4,
                 cls_token: bool = True, channels: int = 3):
        super().__init__()
        self.grid = img_size // patch
        self.patch_embed = PatchEmbed(patch, d, channels)
        self.cls = nn.Parameter(torch.zeros(1, 1, d)) if cls_token else None
        n_tok = self.grid ** 2 + (1 if cls_token else 0)
        self.pos = nn.Parameter(torch.randn(1, n_tok, d) * 0.02)
        self.blocks = nn.ModuleList(Block(d, n_heads, 4 * d) for _ in range(n_layers))  # no mask: bidirectional
        self.norm = RMSNorm(d)

    def forward(self, images):
        x = self.patch_embed(images)                                   # (B, N_p, d)
        if self.cls is not None:
            x = torch.cat([self.cls.expand(x.shape[0], -1, -1), x], 1)  # (B, 1 + N_p, d)
        x = x + self.pos
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)

    def pooled(self, images):
        x = self(images)
        return x[:, 0] if self.cls is not None else x.mean(1)        # (B, d)


def to_tensor(images_uint8, mean=0.5, std=0.25) -> torch.Tensor:
    """(N, H, W, 3) uint8 -> (N, 3, H, W) float, normalised per channel."""
    x = torch.as_tensor(images_uint8).permute(0, 3, 1, 2).float() / 255
    return (x - mean) / std


def resize_pos_embed(pos: torch.Tensor, new_grid: int, n_prefix: int = 1) -> torch.Tensor:
    """Bicubically interpolate learned 2-D position embeddings to a new grid size.

    pos: (1, n_prefix + g*g, d). Used when a ViT is run at a resolution it was not trained on.
    """
    prefix, grid = pos[:, :n_prefix], pos[:, n_prefix:]
    g = int(grid.shape[1] ** 0.5)
    grid = grid.reshape(1, g, g, -1).permute(0, 3, 1, 2)                       # (1, d, g, g)
    grid = F.interpolate(grid, size=(new_grid, new_grid), mode="bicubic", align_corners=False)
    return torch.cat([prefix, grid.permute(0, 2, 3, 1).reshape(1, new_grid ** 2, -1)], 1)
