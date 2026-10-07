"""Action-generation utilities shared by Labs 09 and 11 (book Chapter 6).

Flow matching with linear paths (book 6.5): x_tau = (1 - tau) x0 + tau x1, x0 ~ N(0, I), target velocity x1 - x0,
sampling by Euler integration from tau = 0 (noise) to tau = 1 (data). And a uniform action discretiser for
token-based action heads (book 6.2).
"""

import torch


def fm_loss(v_net, x1, cond):
    """v_net(x_tau, tau, cond) -> velocity with the shape of x1. x1: (B, ...) clean samples."""
    x0 = torch.randn_like(x1)
    tau = torch.rand(len(x1), device=x1.device)
    t = tau.view(-1, *[1] * (x1.dim() - 1))
    x_tau = (1 - t) * x0 + t * x1
    return ((v_net(x_tau, tau, cond) - (x1 - x0)) ** 2).mean()


@torch.no_grad()
def fm_sample(v_net, cond, shape, steps=10, generator=None, device=None):
    """Euler integration of the learned flow from noise; shape = (B, ...)."""
    x = torch.randn(shape, generator=generator, device=device)
    for i in range(steps):
        tau = torch.full((shape[0],), i / steps, device=x.device)
        x = x + v_net(x, tau, cond) / steps
    return x


class Binner:
    """Uniform bins on [low, high] (per dimension); ids in {0..K-1}; decode returns bin centres."""

    def __init__(self, low, high, K=256):
        self.low, self.high, self.K = torch.as_tensor(low).float(), torch.as_tensor(high).float(), K

    def encode(self, a):
        low, high = self.low.to(a.device), self.high.to(a.device)
        x = (a.clamp(low, high) - low) / (high - low)
        return (x * self.K).long().clamp(max=self.K - 1)

    def decode(self, ids):
        low, high = self.low.to(ids.device), self.high.to(ids.device)
        return low + (ids.float() + 0.5) / self.K * (high - low)
