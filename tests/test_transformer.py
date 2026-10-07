import torch

from vlacourse.transformer import TinyGPT


def test_kv_cache_matches_full_recompute():
    torch.manual_seed(0)
    m = TinyGPT(vocab_size=50, d=32, n_layers=2, n_heads=4).eval()
    prompt = torch.randint(0, 50, (2, 5))
    a = m.generate(prompt, 10, temperature=0, use_cache=True)
    b = m.generate(prompt, 10, temperature=0, use_cache=False)
    assert torch.equal(a, b)


def test_causality():
    torch.manual_seed(0)
    m = TinyGPT(vocab_size=50, d=32, n_layers=2, n_heads=4).eval()
    x = torch.randint(0, 50, (1, 8))
    y = x.clone()
    y[0, -1] = (y[0, -1] + 1) % 50  # change only the last token
    with torch.no_grad():
        assert torch.allclose(m(x)[:, :-1], m(y)[:, :-1], atol=1e-6)


def test_initial_loss_near_log_vocab():
    import math
    import torch.nn.functional as F
    torch.manual_seed(0)
    m = TinyGPT(vocab_size=50, d=64, n_layers=2, n_heads=4)
    x = torch.randint(0, 50, (8, 32))
    loss = F.cross_entropy(m(x[:, :-1]).reshape(-1, 50), x[:, 1:].reshape(-1))
    assert abs(loss.item() - math.log(50)) < 0.5
