# %% [markdown]
# # Lab 14 — Inside SmolVLA
#
# **Book sections:** [10.5](../../book/docs/chapters/10-case-studies/groot-smolvla.md),
# [11.1](../../book/docs/chapters/11-deployment/latency.md)–[11.3](../../book/docs/chapters/11-deployment/serving.md),
# [11.5](../../book/docs/chapters/11-deployment/monitoring.md).
#
# **Goal.** Take a real, pretrained VLA apart. Load SmolVLA (≈450 M parameters) with the `lerobot` library,
# feed it a frame from a community LeRobot dataset, and follow the data through every stage: processors,
# normalisation, image and language tokens, the VLM prefix, the flow-matching action expert, the action chunk.
# Then treat it as a deployment engineer would: measure latency per stage at the tail, choose an execution
# horizon, simulate weight quantisation, simulate a policy server shared by a team of robots, and compute an
# uncertainty signal.
#
# **Environment.** `lerobot` pins older `numpy`/`transformers` than the course environment, so this lab runs in a
# separate one: `scripts/setup_lerobot_env.sh`, then `data/venv-lerobot/bin/jupyter lab`.
# **Downloads:** ≈1 GB of weights (SmolVLA + SmolVLM2) and ≈450 MB for one dataset episode (its videos are stored
# in multi-episode files). **GPU:** ≈1.3 GB. **Runtime:** ≈6 min after downloads. **Seed:** 0.

# %%
import hashlib
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from huggingface_hub import try_to_load_from_cache

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, make_att_2d_masks

torch.manual_seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", dev, torch.cuda.get_device_name() if dev == "cuda" else "")

# %% [markdown]
# ## 1. The model
# `lerobot/smolvla_base` is the pretrained (not task-fine-tuned) checkpoint. Count parameters by component and
# compare with book 10.5: SigLIP vision encoder, pixel-shuffle connector, the first 16 of SmolVLM2's 32 language
# layers, and an action expert whose width is 0.75× the VLM's.

# %%
t0 = time.time()
policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base").to(dev).eval()
print(f"loaded in {time.time() - t0:.0f} s, dtype {next(policy.parameters()).dtype}")
cfg, model = policy.config, policy.model
vwe = model.vlm_with_expert
count = lambda mod: sum(p.numel() for p in mod.parameters())  # noqa: E731
parts = {"vision encoder (SigLIP)": vwe.vlm.model.vision_model, "connector (pixel shuffle + MLP)": vwe.vlm.model.connector,
         f"language model ({cfg.num_vlm_layers} layers)": vwe.vlm.model.text_model, "LM head (unused for actions)": vwe.vlm.lm_head,
         "action expert": vwe.lm_expert}
for name, mod in parts.items():
    print(f"{name:34s} {count(mod) / 1e6:7.1f} M")
print(f"{'total':34s} {count(policy) / 1e6:7.1f} M  ({count(policy) * 2 / 1e9:.2f} GB in bf16)")
print(f"\ninputs : {list(cfg.input_features)}\noutputs: {list(cfg.output_features)}")
print(f"chunk {cfg.chunk_size} actions, {cfg.num_steps} flow steps, state/action padded to {cfg.max_state_dim}, "
      f"images resized+padded to {cfg.resize_imgs_with_padding}, expert attention: {cfg.attention_mode} "
      f"(self-attention every {cfg.self_attn_every_n_layers} layers)")

# %% [markdown]
# ## 2. A frame from a community dataset
# `lerobot/svla_so100_pickplace`: a low-cost SO-100 arm picking a cube, two cameras, 30 fps. `delta_timestamps`
# asks the dataset to return, with each frame, the next 50 actions — exactly the training target of a chunked policy.

# %%
H_A = cfg.chunk_size
ds = LeRobotDataset("lerobot/svla_so100_pickplace", episodes=[0],
                    delta_timestamps={"action": [i / 30 for i in range(H_A)]})
print(f"{ds.num_frames} frames at {ds.fps} fps; features: {list(ds.meta.features)}")
item = ds[100]
for k, v in item.items():
    print(f"  {k:28s} {tuple(v.shape) if torch.is_tensor(v) else repr(v)}")
fig, axs = plt.subplots(1, 2, figsize=(9, 3.4))
for ax, k in zip(axs, ["observation.images.top", "observation.images.wrist"]):
    ax.imshow(item[k].permute(1, 2, 0)); ax.set_title(k, fontsize=9); ax.axis("off")
plt.suptitle(item["task"].strip(), fontsize=10); plt.show()

# %% [markdown]
# ## 3. Processors and normalisation statistics
# A LeRobot policy ships with a **preprocessor** (rename cameras, add the batch dimension, tokenise the task,
# move to the device, normalise) and a **postprocessor** (un-normalise, move to the CPU). The base checkpoint's
# camera slots are called `camera1..3`, so we map our cameras onto them. First, use the processors exactly as
# shipped:

# %%
RENAME = {"observation.images.top": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}
KEYS = ["observation.images.top", "observation.images.wrist", "observation.state", "task"]
pre0, post0 = make_pre_post_processors(cfg, pretrained_path="lerobot/smolvla_base",
                                       preprocessor_overrides={"rename_observations_processor": {"rename_map": RENAME}})
print("preprocessor :", [type(s).__name__ for s in pre0.steps])
print("postprocessor:", [type(s).__name__ for s in post0.steps])
obs = {k: item[k] for k in KEYS}
noise = torch.randn(1, H_A, cfg.max_action_dim, generator=torch.Generator().manual_seed(0)).to(dev)
policy.reset()
a0 = post0(policy.predict_action_chunk(pre0(obs), noise=noise))[0].cpu()
print("first predicted action:", a0[0].numpy().round(2))
print("recorded action       :", item["action"][0].numpy().round(2))

# %% [markdown]
# The prediction is in **normalised** units while the recorded actions are joint angles in degrees: the base
# checkpoint carries no statistics for this robot, so its (un)normaliser is the identity. Normalisation
# statistics belong to the *dataset and robot*, and must travel with every checkpoint (book 8.3, 9.3). Plug in the
# dataset's:

# %%
STATS = ds.meta.stats
pre, post = make_pre_post_processors(
    cfg, pretrained_path="lerobot/smolvla_base",
    preprocessor_overrides={"rename_observations_processor": {"rename_map": RENAME}, "normalizer_processor": {"stats": STATS}},
    postprocessor_overrides={"unnormalizer_processor": {"stats": STATS}})


def predict(o, noise=noise, steps=None):
    policy.reset()
    if steps is not None:
        policy.config.num_steps = steps
    out = post(policy.predict_action_chunk(pre(o), noise=noise)).cpu()
    policy.config.num_steps = 10
    return out


batch = pre(obs)
for k, v in batch.items():
    if torch.is_tensor(v):
        print(f"  {k:38s} {str(tuple(v.shape)):18s} {v.dtype}")
pred = predict(obs)[0]
print("first predicted action:", pred[0].numpy().round(1), "\nrecorded action       :", item["action"][0].numpy().round(1))

# %%
names = ds.meta.features["action"]["names"]
names = names if isinstance(names, list) else [f"joint {i}" for i in range(6)]
fig, axs = plt.subplots(1, 3, figsize=(12, 3))
for ax, j in zip(axs, [0, 1, 5]):
    ax.plot(item["action"][:, j], label="recorded next 50 actions")
    ax.plot(pred[:, j], "--", label="SmolVLA chunk")
    ax.set_title(str(names[j]), fontsize=9); ax.set_xlabel("step in chunk (1/30 s)")
axs[0].set_ylabel("degrees"); axs[0].legend(fontsize=8); plt.tight_layout(); plt.show()

errs = []
for t in range(0, ds.num_frames - H_A, 25):
    it = ds[t]
    errs.append((predict({k: it[k] for k in KEYS})[0] - it["action"]).abs().mean(0))
errs = torch.stack(errs)
std = torch.as_tensor(STATS["action"]["std"]).float()
print(f"mean |chunk error| over {len(errs)} frames, per joint (degrees):", errs.mean(0).numpy().round(1))
print("action std per joint (degrees)                   :", std.numpy().round(1))

# %% [markdown]
# The base model's chunks follow the recorded motion roughly — errors are a fraction of each joint's spread —
# without any fine-tuning on this dataset. A fine-tuned checkpoint would do much better; that is what
# post-training is for (book 9.1).
#
# ## 4. Counting tokens
# Each camera image is resized and padded to 512×512, encoded by SigLIP into 32×32 patches, and reduced by the
# connector's pixel shuffle to 64 tokens. Then come the instruction's tokens (padded to 48) and one state token.

# %%
with torch.no_grad():
    images, img_masks = policy.prepare_images(batch)
    state = policy.prepare_state(batch)
    per_cam = vwe.embed_image(images[0]).shape[1]
    emb, pad, att = model.embed_prefix(images, img_masks, batch["observation.language.tokens"],
                                       batch["observation.language.attention_mask"], state=state)
n_lang = int(batch["observation.language.attention_mask"].sum())
n_special = int(pad.sum()) - 2 * per_cam - n_lang - 1
print(f"prepared images: {[tuple(i.shape) for i in images]}")
print(f"prefix: {emb.shape[1]} positions, {int(pad.sum())} real = 2 cameras x {per_cam} image tokens + {n_lang} "
      f"instruction tokens + 1 state token + {n_special} image start/end tokens; width {emb.shape[-1]}")
print(f"action expert input: {H_A} action tokens of width {vwe.expert_hidden_size}")

# %% [markdown]
# ## 5. Latency, stage by stage (book 11.1)
# Time each stage 30 times at batch 1 and report the median, p90, p99 and maximum. Note LeRobot's convention:
# flow time runs from 1 (noise) to 0 (actions), the reverse of the book's; the integration is the same.

# %%
class StageTimer:
    def __init__(self):
        self.samples = defaultdict(list)

    def measure(self, stage, fn):
        t0 = time.perf_counter()
        out = fn()
        if dev == "cuda":
            torch.cuda.synchronize()
        self.samples[stage].append((time.perf_counter() - t0) * 1000)
        return out

    def report(self):
        for stage, v in self.samples.items():
            v = np.array(v)
            print(f"{stage:22s} p50 {np.percentile(v, 50):7.1f}  p90 {np.percentile(v, 90):7.1f}  "
                  f"p99 {np.percentile(v, 99):7.1f}  max {v.max():7.1f} ms")


def one_chunk(T, steps=10):
    b = T.measure("preprocess", lambda: pre(obs))
    ims, ms = T.measure("prepare images", lambda: policy.prepare_images(b))
    st = policy.prepare_state(b)
    e, p_, a_ = T.measure("vision + embed prefix", lambda: model.embed_prefix(
        ims, ms, b["observation.language.tokens"], b["observation.language.attention_mask"], state=st))
    _, kv = T.measure("VLM prefill (cached)", lambda: vwe.forward(
        attention_mask=make_att_2d_masks(p_, a_), position_ids=torch.cumsum(p_, 1) - 1, past_key_values=None,
        inputs_embeds=[e, None], use_cache=True))

    def expert():
        x = noise.clone()
        for i in range(steps):
            x = x - model.denoise_step(p_, kv, x, torch.full((1,), 1 - i / steps, device=dev)) / steps
        return x
    x = T.measure(f"action expert x {steps}", expert)
    return T.measure("postprocess", lambda: post(x[:, :, :6].float()))


with torch.no_grad():
    for _ in range(3):
        one_chunk(StageTimer())                                     # warm-up
    T = StageTimer()
    for _ in range(30):
        one_chunk(T)
T.report()
total = np.sum([np.array(v) for v in T.samples.values()], axis=0)
p99 = np.percentile(total, 99)
print(f"\nend-to-end p50 {np.median(total):.0f} ms, p99 {p99:.0f} ms")
for hz in (10, 30, 50):
    print(f"at {hz:2d} Hz a synchronous executor needs H_e >= {math.ceil(p99 / 1000 * hz):3d} actions per chunk "
          f"({'fits' if math.ceil(p99 / 1000 * hz) <= H_A else 'exceeds'} the {H_A}-action chunk)")

# %% [markdown]
# Compare the split with book 6.6 and 8.7: the vision encoder and the prefix run once per chunk; the expert runs
# `num_steps` times. In the committed run (a 6 GB laptop GPU) the ten expert passes take about 70% of the 126 ms
# total — more than SigLIP and the cached prefill together. Each pass is small (50 action tokens) but goes through
# 16 expert layers at batch 1, where kernel launches and Python dominate the arithmetic. The cost model of book 6.6
# predicts *which* parts scale with what; only a measurement tells you which part dominates on your hardware.
# Reducing flow steps trades accuracy for time — measure how far the chunk moves:

# %%
ref = predict(obs, steps=10)[0]
for steps in (1, 2, 4, 10, 20):
    with torch.no_grad():
        Ts = StageTimer()
        for _ in range(5):
            Ts.measure("chunk", lambda: predict(obs, steps=steps))
    dif = (predict(obs, steps=steps)[0] - ref).abs().mean().item()
    print(f"{steps:2d} flow steps: {np.median(Ts.samples['chunk']):6.0f} ms per chunk, mean |change| vs 10 steps {dif:5.2f} deg")

# %% [markdown]
# ## 6. Weight quantisation, simulated (book 11.2)
# Quantise every linear weight of the language model and the action expert, group-wise (64 weights per scale),
# to int8 and int4, then dequantise back to bf16 ("fake quantisation"). This shows the *accuracy* effect with the
# same noise; it gives no speed-up — that needs low-bit kernels — and the memory column is computed, not measured.

# %%
def fake_quant_(w, bits, group=64):
    q = 2 ** (bits - 1) - 1
    g = w.float().reshape(-1, group)
    s = g.abs().amax(1, keepdim=True).clamp(min=1e-8) / q
    w.copy_((torch.clamp(torch.round(g / s), -q - 1, q) * s).reshape(w.shape).to(w.dtype))


linears = [m for root in (vwe.vlm.model.text_model, vwe.lm_expert) for m in root.modules() if isinstance(m, torch.nn.Linear)]
backup = [l.weight.detach().cpu().clone() for l in linears]
n_q = sum(l.weight.numel() for l in linears)
n_rest = count(policy) - n_q - count(vwe.vlm.lm_head)
for bits in (16, 8, 4):
    if bits < 16:
        with torch.no_grad():
            for l, b in zip(linears, backup):
                l.weight.copy_(b.to(l.weight.device)); fake_quant_(l.weight, bits)
    mem = (n_q * (bits / 8 + (2 / 64 if bits < 16 else 0)) + n_rest * 2) / 1e9
    chunks = torch.stack([predict({k: ds[t][k] for k in KEYS})[0] for t in (0, 100, 200, 300)])
    if bits == 16:
        ref_chunks = chunks
    print(f"{'bf16' if bits == 16 else f'int{bits}':5s} weights ≈ {mem:.2f} GB; mean |chunk change| vs bf16: "
          f"{(chunks - ref_chunks).abs().mean():.2f} deg")
with torch.no_grad():
    for l, b in zip(linears, backup):
        l.weight.copy_(b.to(l.weight.device))

# %% [markdown]
# **Reading §6.** int8 moves the chunks by a fraction of a degree, int4 by a few degrees — small next to the
# model's own error on this dataset (§3) and next to the spread between noise samples (§8), but not negligible for
# precise tasks. Validate in closed loop before deploying int4 (book 11.2).
#
# ## 7. A policy server for a team (book 11.3)
# Measure the time for one chunk at batch sizes 1–8, then simulate $N$ robots that each execute $H_e=8$ actions
# at 10 Hz (a new chunk every 0.8 s, requested when 4 actions remain, so a response has 0.4 s before the robot
# runs dry). The server waits up to a batching window $w$ for more requests, then serves them all in one pass.

# %%
def batch_of(B):
    b = pre(obs)
    out = {k: v.repeat(B, *[1] * (v.dim() - 1)) for k, v in b.items() if torch.is_tensor(v)}
    out["task"] = b["task"] * B
    return out


t_batch = {}
with torch.no_grad():
    for B in (1, 2, 4, 8):
        bb, ts = batch_of(B), []
        for _ in range(5):
            policy.reset(); t0 = time.perf_counter(); policy.predict_action_chunk(bb)
            if dev == "cuda":
                torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
        t_batch[B] = float(np.median(ts))
print({B: f"{t * 1000:.0f} ms" for B, t in t_batch.items()})
tb = lambda B: float(np.interp(B, list(t_batch), list(t_batch.values())))  # noqa: E731


def simulate(n_robots, window, period=0.8, slack=0.4, horizon=60.0, seed=0):
    rng = np.random.default_rng(seed)
    nxt = list(rng.uniform(0, period, n_robots))            # next request time of each robot
    t, waits, busy = 0.0, [], 0.0
    while t < horizon:
        first = min(nxt)
        t = max(t, first)
        cutoff = t + window                                   # batch everything that arrives within the window
        group = [i for i in range(n_robots) if nxt[i] <= cutoff][:8]
        start = max(t, max(nxt[i] for i in group))
        dt = tb(len(group))
        done = start + dt
        busy += dt
        for i in group:
            waits.append(done - nxt[i])
            nxt[i] += period
        t = done
    w = np.array(waits)
    return w, (w > slack).mean(), busy / t


fig, ax = plt.subplots(figsize=(6.5, 3))
for window in (0.0, 0.05, 0.15):
    p99s = []
    for n in range(1, 9):
        w, miss, util = simulate(n, window)
        p99s.append(np.percentile(w, 99))
        if window == 0.05:
            print(f"{n} robots, window 50 ms: request→chunk p50 {np.median(w) * 1000:4.0f} ms, p99 "
                  f"{np.percentile(w, 99) * 1000:4.0f} ms, missed {miss:5.1%}, GPU busy {util:4.0%}")
    ax.plot(range(1, 9), np.array(p99s) * 1000, "o-", label=f"window {window * 1000:.0f} ms")
ax.axhline(400, color="red", ls="--", lw=1); ax.text(1, 410, "slack: 4 actions at 10 Hz", color="red", fontsize=8)
ax.set_xlabel("robots served"); ax.set_ylabel("p99 request→chunk (ms)"); ax.legend(fontsize=8); plt.show()

# %% [markdown]
# **Reading §7.** A chunk for 8 robots costs about four times one for a single robot, not eight: batching helps,
# but less than the bandwidth argument of book 11.3 suggests, because two 512-pixel images per robot make the
# prefix compute-heavy. In the simulation one laptop GPU serves 8 robots without missed deadlines, at high load,
# and with a p99 latency close to the 0.4 s slack — the next robot, a slower GPU or a longer prefix would break it.
#
# ## 8. An uncertainty signal and a log record (book 11.5)
# Sample 8 chunks with different noise for the same observation and measure their spread, at a few frames of the
# episode. Then write the record a deployment would log for each chunk.

# %%
spread = []
for t in range(0, ds.num_frames - H_A, 50):
    o = {k: ds[t][k] for k in KEYS}
    cs = torch.stack([predict(o, noise=torch.randn(1, H_A, cfg.max_action_dim, generator=torch.Generator().manual_seed(s)).to(dev))[0]
                      for s in range(8)])
    spread.append((t, cs.std(0).mean().item()))
print("frame : mean std over 8 sampled chunks (deg)")
for t, s in spread:
    print(f"{t:5d} : {s:5.2f}")
ckpt = try_to_load_from_cache("lerobot/smolvla_base", "model.safetensors")
weights_hash = hashlib.sha256(Path(ckpt).read_bytes()).hexdigest()[:16] if isinstance(ckpt, str) else "unknown"
record = {"t_obs": float(item["timestamp"]), "task": item["task"].strip(), "model": "lerobot/smolvla_base",
          "weights_sha256": weights_hash, "stats": "svla_so100_pickplace meta/stats.json",
          "num_steps": cfg.num_steps, "chunk": pred[:2].numpy().round(2).tolist(), "latency_ms_p50": float(np.median(total))}
print(json.dumps(record)[:400], "…")

# %% [markdown]
# ## Exercises
# 1. **Fine-tune.** With the `lerobot` training script, fine-tune `smolvla_base` on a few episodes of this
#    dataset (book 9.4) and repeat §3. How much does the chunk error drop?
# 2. **Fewer image tokens.** Feed only the top camera. How do the prefix length, latency and the chunk change?
# 3. **Asynchronous execution.** Using §5's p99 and 30 Hz control, compute the staleness of the first new action
#    with asynchronous execution, and the smallest refill threshold that never starves the controller.
# 4. **Real quantisation.** If `bitsandbytes` supports your GPU, load the language model in 8-bit and measure
#    memory and latency; compare with §6.
# 5. **Server policy.** Replace "first come, first served" in `simulate` by "most urgent first" (smallest
#    remaining queue). When does it help?
