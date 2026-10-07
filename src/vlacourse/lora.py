"""Low-rank adaptation (LoRA; book 9.4): W x  ->  W x + (alpha / r) B A x, with W frozen, A random, B = 0."""

import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float | None = None):
        super().__init__()
        self.base, self.r, self.scale = base, r, (alpha or r) / r
        self.A = nn.Parameter(torch.empty(r, base.in_features, device=base.weight.device))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, device=base.weight.device))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        base.weight.requires_grad = False
        if base.bias is not None:
            base.bias.requires_grad = False

    def forward(self, x):
        return self.base(x) + (x @ self.A.T @ self.B.T) * self.scale

    @torch.no_grad()
    def merged(self) -> nn.Linear:
        """A plain Linear with the update folded in: no extra cost at inference."""
        self.base.weight += self.scale * self.B @ self.A
        return self.base


def add_lora(module: nn.Module, r: int, alpha: float | None = None) -> list[nn.Parameter]:
    """Replace every nn.Linear inside `module` by a LoRALinear; return the new trainable parameters."""
    params = []
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            lora = LoRALinear(child, r, alpha)
            setattr(module, name, lora)
            params += [lora.A, lora.B]
        else:
            params += add_lora(child, r, alpha)
    return params


def merge_lora(module: nn.Module) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, LoRALinear):
            setattr(module, name, child.merged())
        else:
            merge_lora(child)
