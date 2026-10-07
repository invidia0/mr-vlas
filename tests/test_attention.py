import torch
import torch.nn.functional as F

from vlacourse.attention import causal_mask, scaled_dot_product_attention


def test_matches_pytorch_sdpa():
    torch.manual_seed(0)
    q, k, v = torch.randn(2, 3, 5, 8), torch.randn(2, 3, 7, 8), torch.randn(2, 3, 7, 4)
    out, w = scaled_dot_product_attention(q, k, v)
    assert torch.allclose(out, F.scaled_dot_product_attention(q, k, v), atol=1e-5)
    assert torch.allclose(w.sum(-1), torch.ones(2, 3, 5))


def test_causal_mask_blocks_future():
    torch.manual_seed(0)
    q = k = v = torch.randn(1, 4, 6, 8)
    out, w = scaled_dot_product_attention(q, k, v, causal_mask(6))
    assert torch.all(w.triu(1) == 0)
    assert torch.allclose(out, F.scaled_dot_product_attention(q, k, v, is_causal=True), atol=1e-5)
