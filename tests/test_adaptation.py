import numpy as np
import torch
import torch.nn as nn

from vlacourse.depot_env import DepotNav, evaluate
from vlacourse.lora import add_lora, merge_lora


def test_lora_starts_identical_and_merges():
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 8))
    x = torch.randn(5, 16)
    before = net(x)
    params = add_lora(net, r=4)
    assert torch.allclose(net(x), before)                      # B = 0: no change at initialisation
    assert sum(p.numel() for p in params) == 4 * (16 + 32) + 4 * (32 + 8)
    assert all(not p.requires_grad for n, p in net.named_parameters() if "base" in n)
    with torch.no_grad():
        for p in params:
            p.add_(0.1 * torch.randn_like(p))
    adapted = net(x)
    merge_lora(net)
    assert all(isinstance(m, nn.Linear) for m in (net[0], net[2]))
    assert torch.allclose(net(x), adapted, atol=1e-5)


def test_depot_kinds_and_targets():
    assert DepotNav().reset(seed=10)["instruction"] == "go to the blue cone"   # default episodes unchanged
    env = DepotNav(kinds=("toolbox", "crate", "barrel"))
    for s in range(50):
        env.reset(seed=s)
        assert all(o.kind != "cone" for o in env.scene.objects)
    env = DepotNav(target_kinds=("cone",))
    for s in range(50):
        o = env.reset(seed=s)
        assert "cone" in o["instruction"] and all(t.kind == "cone" for t in env.targets)


def test_evaluate_groups_match_lockstep():
    pol = lambda im, ins, st: np.zeros((len(im), 8, 2), np.float32)  # noqa: E731
    a, b = evaluate(pol, 6), evaluate(pol, 6, parallel=4)
    assert (a["steps"] == b["steps"]).all() and (a["ambiguous"] == b["ambiguous"]).all()


def test_cbf_keeps_robot_out_of_zone_and_inside_fence():
    from vlacourse.safety import cbf_filter
    zone, fence = (np.array([0.5, 0.5]), 0.15), (np.zeros(2), np.ones(2))
    x, dt = np.array([0.05, 0.38]), 0.02
    u, active = cbf_filter(np.array([0.05, 0.9]), np.array([0.3, 0.0]), [zone], fence)
    assert not active                                       # far from everything: untouched
    for _ in range(300):                                    # head through the zone (off-centre), then out of the box
        u, _ = cbf_filter(x, np.array([1.0, 0.05]), [zone], fence, u_max=1.0)
        x = x + dt * u
        assert np.linalg.norm(x - zone[0]) >= zone[1] - 1e-3
        assert (x >= -1e-3).all() and (x <= 1 + 1e-3).all()
    assert x[0] > 0.9                                       # it went around, not stopped


def test_cbf_separation_and_watchdog():
    from vlacourse.safety import Watchdog, cbf_filter
    a, b, dt = np.array([0.0, 0.0]), np.array([1.0, 0.01]), 0.02
    for _ in range(200):                                    # two robots driving head-on
        ua, _ = cbf_filter(a, b - a, others=[b], sep=0.2)
        ub, _ = cbf_filter(b, a - b, others=[a], sep=0.2)
        a, b = a + dt * ua, b + dt * ub
        assert np.linalg.norm(a - b) >= 0.2 - 1e-3
    wd = Watchdog(timeout=0.3, escalate=1.0)
    wd.feed(0.0)
    assert not wd(0.1, [1, 0])[1] and wd(0.5, [1, 0])[1]
    wd.feed(0.6)
    assert not wd(0.7, [1, 0])[1]
    assert wd(2.0, [1, 0])[1] and (wd.feed(2.1) or wd(2.2, [1, 0])[1])   # failsafe latched


def test_auction_is_optimal_and_greedy_is_not_always():
    from vlacourse.teams import auction, brute_force, greedy, total
    rng = np.random.default_rng(0)
    worse = 0
    for _ in range(50):
        c = rng.uniform(0, 1, (4, 5))
        a, _ = auction(c, eps=1e-6)
        assert len(set(a)) == 4 and abs(total(c, a) - total(c, brute_force(c))) < 4e-6
        worse += total(c, greedy(c)) > total(c, brute_force(c)) + 1e-9
    assert worse > 0


def test_decentralised_auction_without_loss_is_conflict_free():
    from vlacourse.teams import LossyChannel, decentralised_auction
    rng = np.random.default_rng(1)
    for s in range(20):
        c = rng.uniform(0, 1, (3, 3))
        mine, _ = decentralised_auction(c, LossyChannel(3, p_loss=0.0, seed=s), np.zeros((3, 2)), rounds=40)
        assert sorted(mine) == [0, 1, 2]
