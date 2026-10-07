import random

import torch

from vlacourse.clip import TinyCLIP, clip_loss, siglip_loss
from vlacourse.data import scenes
from vlacourse.transformer import TinyGPT
from vlacourse.vision import ViT, resize_pos_embed
from vlacourse.vlm import TinyVLM


def test_scenes_deterministic_and_answerable():
    a, ia = scenes.scene_dataset(4, seed=3)
    b, ib = scenes.scene_dataset(4, seed=3)
    assert a == b and (ia == ib).all() and ia.shape == (4, 64, 64, 3)
    for s in a:
        for q, ans in scenes.qa_pairs(s):
            assert scenes.decode(scenes.encode(q)) == q and scenes.encode(ans)


def test_vit_shapes_and_pos_resize():
    vit = ViT(img_size=64, patch=8, d=32, n_layers=1, n_heads=4)
    assert vit(torch.randn(2, 3, 64, 64)).shape == (2, 1 + 64, 32)
    assert resize_pos_embed(vit.pos, 12).shape == (1, 1 + 144, 32)


def test_contrastive_losses_at_chance():
    torch.manual_seed(0)
    B = 16
    e = torch.nn.functional.normalize(torch.randn(B, 8), dim=-1)
    # identical, uninformative embeddings: every caption is equally likely -> loss = log B
    same = torch.ones(B, 8) / 8 ** 0.5
    assert abs(clip_loss(same, same, torch.tensor(0.0)).item() - torch.log(torch.tensor(B * 1.0)).item()) < 1e-5
    assert clip_loss(e, e, torch.tensor(4.6)) < clip_loss(e, e.roll(1, 0), torch.tensor(4.6))
    assert siglip_loss(e, e, torch.tensor(2.3), torch.tensor(-10.0)) > 0


def test_tiny_clip_embeddings_normalised():
    m = TinyCLIP(len(scenes.VOCAB), d=32, d_e=16)
    ids = torch.tensor([scenes.encode("a red toolbox") + [scenes.EOS]])
    t = m.encode_text(ids, torch.tensor([3]))
    i = m.encode_image(torch.randn(1, 3, 64, 64))
    assert torch.allclose(t.norm(dim=-1), torch.ones(1)) and torch.allclose(i.norm(dim=-1), torch.ones(1))


def test_vlm_cached_generation_matches_full_recompute():
    torch.manual_seed(0)
    for prefix_lm in (False, True):
        vlm = TinyVLM(ViT(32, 8, 32, 1, 4), TinyGPT(len(scenes.VOCAB), d=32, n_layers=2, n_heads=4), 32, prefix_lm).eval()
        img, ids = torch.randn(2, 3, 32, 32), torch.randint(4, 20, (2, 5))
        gen = vlm.generate(img, ids, 6)
        seq = ids
        for _ in range(6):  # recompute everything from scratch each step
            seq = torch.cat([seq, vlm(img, seq)[:, -1:].argmax(-1)], 1)
        assert torch.equal(gen, seq[:, 5:])


def test_binner_roundtrip_and_flow_matching_learns_bimodal():
    from vlacourse.generative import Binner, fm_loss, fm_sample
    b = Binner([-1.0, -1.0], [1.0, 1.0], K=256)
    a = torch.rand(100, 2) * 2 - 1
    assert (b.decode(b.encode(a)) - a).abs().max() <= 1 / 256 + 1e-6
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Linear(2, 64), torch.nn.SiLU(), torch.nn.Linear(64, 64), torch.nn.SiLU(),
                              torch.nn.Linear(64, 1))
    v = lambda x, tau, c: net(torch.cat([x, tau[:, None]], -1))  # noqa: E731
    opt = torch.optim.Adam(net.parameters(), 3e-3)
    for _ in range(1500):
        x1 = (torch.randint(0, 2, (256, 1)) * 2 - 1).float() + 0.05 * torch.randn(256, 1)
        loss = fm_loss(v, x1, None)
        opt.zero_grad(); loss.backward(); opt.step()
    s = fm_sample(v, None, (2000, 1), steps=20)
    assert (s.abs() < 0.5).float().mean() < 0.1          # few samples between the modes at +-1
    assert 0.3 < (s > 0).float().mean() < 0.7             # both modes used


def test_tiny_vla_heads_shapes_and_losses():
    from vlacourse.depot_env import INSTR_VOCAB
    from vlacourse.vla import TinyVLA
    torch.manual_seed(0)
    img, instr, state = torch.randn(2, 3, 64, 64), torch.tensor([[40, 41, 30, 0], [40, 41, 30, 35]]), torch.rand(2, 2)
    chunk = torch.rand(2, 8, 2) * 2 - 1
    for head in ("mse", "tokens", "flow"):
        m = TinyVLA(len(INSTR_VOCAB), head=head, d=32, n_layers=1, d_vision=32,
                    vision=ViT(64, 8, 32, n_layers=1, n_heads=4)).eval()
        assert m.loss(img, instr, state, chunk).item() > 0
        assert m.act(img, instr, state, steps=2).shape == (2, 8, 2)


def test_token_head_prefix_never_reads_actions():
    from vlacourse.depot_env import INSTR_VOCAB
    from vlacourse.vla import TinyVLA
    torch.manual_seed(0)
    m = TinyVLA(len(INSTR_VOCAB), head="tokens", d=32, n_layers=2, d_vision=32,
                vision=ViT(64, 8, 32, n_layers=1, n_heads=4)).eval()
    img, instr, state = torch.randn(1, 3, 64, 64), torch.tensor([[40, 41, 30, 35]]), torch.rand(1, 2)
    a = torch.tensor([[32, 3, 7, 9]])
    b = a.clone(); b[0, -1] = 1                                  # change only the last action token
    with torch.no_grad():
        la, lb = m._token_logits(img, instr, state, a), m._token_logits(img, instr, state, b)
    assert torch.allclose(la[:, :-1], lb[:, :-1], atol=1e-5)    # earlier predictions unaffected (causal)
