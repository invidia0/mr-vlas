"""A 2-D depot navigation task: the environment of the toy VLA (Lab 11) and later chapters.

A robot (or a drone flying low) sits on the depot floor among a few objects and receives an instruction
such as "go to the red toolbox". It observes a top-down image with itself drawn in it, its own position
(proprioception), and the instruction; it acts with a 2-D velocity command in [-1, 1]^2 (norm <= 1).

Instructions can be **ambiguous**: "go to the red object" when two objects are red. The scripted expert
then picks one target at random per episode, so the demonstrations are genuinely multimodal — the case
that breaks regression policies (book 0.4, Chapter 6).
"""

import random
import re
from dataclasses import dataclass, field

import numpy as np

from .data import scenes

INSTR_VOCAB = scenes.VOCAB + ["go", "to"]    # instruction words appended, so scene-vocabulary ids are unchanged
_INDEX = {w: i for i, w in enumerate(INSTR_VOCAB)}


def encode_instruction(text: str) -> list[int]:
    return [_INDEX[w] for w in re.findall(r"[a-z]+|[,?]", text.lower())]


@dataclass
class DepotNav:
    """Kinematic point robot: p_{t+1} = clip(p_t + v_step * a_t), a_t in the unit disc."""

    res: int = 64
    v_step: float = 0.04            # distance per step at full speed (the image is 1 x 1)
    max_steps: int = 50
    success_radius: float = 0.07
    p_ambiguous: float = 0.3        # probability of a colour-only or kind-only instruction
    kinds: tuple = tuple(scenes.KINDS)   # object kinds that may appear (Lab 13 holds one out)
    target_kinds: tuple | None = None    # if set, instructions only name objects of these kinds
    rng: random.Random = field(default_factory=lambda: random.Random(0))

    def reset(self, seed: int | None = None, keep_scene: bool = False):
        """New episode. keep_scene=True keeps the objects and draws a new instruction and start position —
        so that the same scene appears with different instructions (otherwise a policy can memorise
        scene -> target and ignore the language)."""
        if seed is not None:
            self.rng = random.Random(seed)
        r = self.rng
        while True:
            if not keep_scene:
                self.scene = scenes.random_scene(r, n_objects=(2, 3), min_dist=0.32, kinds=self.kinds)
            objs = self.scene.objects
            cands = [q for q in objs if self.target_kinds is None or q.kind in self.target_kinds]
            if not cands:
                keep_scene = False
                continue
            mode = "full" if r.random() > self.p_ambiguous else r.choice(["color", "kind"])
            if self.target_kinds is not None and mode == "color":
                mode = "full"                            # a colour-only instruction could name other kinds
            o = r.choice(cands)
            if mode == "full":
                self.instruction = f"go to the {o.color} {o.kind}"
                self.targets = [o]
            elif mode == "color":
                self.instruction = f"go to the {o.color} object"
                self.targets = [q for q in objs if q.color == o.color]
            else:
                self.instruction = f"go to the {o.kind}"
                self.targets = [q for q in objs if q.kind == o.kind]
            # start somewhere free, at least 0.3 from every target
            for _ in range(100):
                p = np.array([r.uniform(0.08, 0.92), r.uniform(0.08, 0.92)])
                if all(np.hypot(*(p - [q.x, q.y])) > 0.3 for q in self.targets) and \
                   all(np.hypot(*(p - [q.x, q.y])) > 0.12 for q in objs):
                    break
            else:
                continue
            self.p, self.t = p, 0
            self.goal = r.choice(self.targets)          # the expert's choice when ambiguous
            self.seed = r.randrange(1 << 30)
            return self.obs()

    def obs(self):
        img = scenes.render(self.scene, self.res, seed=self.seed + self.t, agents=[(self.p[0], self.p[1])])
        return {"image": img, "state": self.p.astype(np.float32).copy(), "instruction": self.instruction}

    def distance_to_targets(self):
        return min(np.hypot(*(self.p - [q.x, q.y])) for q in self.targets)

    def step(self, action):
        a = np.asarray(action, dtype=np.float64)
        n = np.linalg.norm(a)
        if n > 1:
            a = a / n
        self.p = np.clip(self.p + self.v_step * a, 0.0, 1.0)
        self.t += 1
        success = self.distance_to_targets() < self.success_radius
        done = success or self.t >= self.max_steps
        return self.obs(), done, success

    def expert_action(self, noise: float = 0.0):
        """Head for the chosen goal at full speed, slowing down in the last step."""
        d = np.array([self.goal.x, self.goal.y]) - self.p
        a = d / self.v_step
        n = np.linalg.norm(a)
        if n > 1:
            a = a / n
        if noise:
            a = a + noise * np.array([self.rng.gauss(0, 1), self.rng.gauss(0, 1)])
        return a.astype(np.float32)


def collect(n_episodes: int, seed: int = 0, chunk: int = 8, noise: float = 0.0, env: DepotNav | None = None,
            tasks_per_scene: int = 1):
    """Expert demonstrations; each scene is reused for `tasks_per_scene` consecutive episodes with different
    instructions and starts. Returns a dict of arrays with one row per time step:

    image (N, res, res, 3) uint8, state (N, 2), action (N, 2), chunk (N, chunk, 2) — the next `chunk`
    expert actions, padded with zeros after the episode ends — instruction ids (list), episode (N,).
    """
    env = env or DepotNav()
    env.reset(seed)
    out = {k: [] for k in ("image", "state", "action", "chunk", "instruction", "episode")}
    for ep in range(n_episodes):
        obs = env.reset(seed) if ep == 0 else env.reset(keep_scene=ep % tasks_per_scene != 0)
        ids = encode_instruction(obs["instruction"])
        ep_acts, ep_rows = [], []
        done = False
        while not done:
            a = env.expert_action(noise)
            ep_rows.append((obs["image"], obs["state"]))
            ep_acts.append(a)
            obs, done, _ = env.step(a)
        ep_acts = np.array(ep_acts)
        padded = np.concatenate([ep_acts, np.zeros((chunk, 2), np.float32)])
        for t, (img, st) in enumerate(ep_rows):
            out["image"].append(img)
            out["state"].append(st)
            out["action"].append(ep_acts[t])
            out["chunk"].append(padded[t:t + chunk])
            out["instruction"].append(ids)
            out["episode"].append(ep)
    return {k: (v if k == "instruction" else np.array(v)) for k, v in out.items()}


def pad_instructions(id_lists, length=None):
    length = length or max(map(len, id_lists))
    out = np.zeros((len(id_lists), length), dtype=np.int64)
    for i, ids in enumerate(id_lists):
        out[i, :len(ids)] = ids
    return out


def evaluate(policy, n_episodes=100, seed=10_000, execute=1, ensemble_m=None, max_batch=256, env_factory=DepotNav,
             parallel=None):
    """Closed-loop evaluation of a chunking policy on DepotNav, all episodes stepped in lockstep.

    policy(images uint8 (B, H, W, 3), instr int64 (B, L), state float32 (B, 2)) -> chunks (B, H_a, 2).
    execute: actions executed open loop from each chunk before re-querying (ignored with ensembling).
    ensemble_m: if set, re-query every step and execute the exponentially weighted average of all chunk
    predictions for the current step, w_i = exp(-m * i) with i = 0 the oldest prediction (ACT).
    env_factory: builds an environment (e.g. MujocoDepotNav); parallel: how many episodes run in lockstep
    (default all; MujocoDepotNav shares one renderer, so it needs parallel=1).
    Returns a dict with per-episode success, steps, ambiguity and the number of policy calls.
    """
    parallel = parallel or n_episodes
    if parallel < n_episodes:
        parts = [evaluate(policy, min(parallel, n_episodes - k), seed + k, execute, ensemble_m, max_batch, env_factory)
                 for k in range(0, n_episodes, parallel)]
        out = {key: np.concatenate([r[key] for r in parts]) for key in ("success", "steps", "ambiguous")}
        out["calls"] = sum(r["calls"] for r in parts)
        return out
    envs = [env_factory() for _ in range(n_episodes)]
    obs = [env.reset(seed + i) for i, env in enumerate(envs)]
    instr = pad_instructions([encode_instruction(o["instruction"]) for o in obs])
    done = np.zeros(n_episodes, bool)
    success = np.zeros(n_episodes, bool)
    steps = np.zeros(n_episodes, int)
    queue = [[] for _ in range(n_episodes)]          # pending actions (open-loop execution)
    preds = [dict() for _ in range(n_episodes)]       # step -> list of predicted actions (ensembling)
    calls = 0
    while not done.all():
        need = [i for i in range(n_episodes) if not done[i] and (ensemble_m is not None or not queue[i])]
        for k in range(0, len(need), max_batch):
            idx = need[k:k + max_batch]
            chunks = policy(np.stack([obs[i]["image"] for i in idx]), instr[idx],
                            np.stack([obs[i]["state"] for i in idx]))
            calls += 1
            for j, i in enumerate(idx):
                if ensemble_m is None:
                    queue[i] = list(chunks[j][:execute])
                else:
                    for h, a in enumerate(chunks[j]):
                        preds[i].setdefault(steps[i] + h, []).append(a)
        for i in range(n_episodes):
            if done[i]:
                continue
            if ensemble_m is None:
                a = queue[i].pop(0)
            else:
                p = preds[i].pop(steps[i])                                   # oldest first
                w = np.exp(-ensemble_m * np.arange(len(p)))
                a = (w[:, None] * np.array(p)).sum(0) / w.sum()
            obs[i], d, s = envs[i].step(a)
            steps[i] += 1
            done[i], success[i] = d, s
    ambiguous = np.array([len(env.targets) > 1 for env in envs])
    return {"success": success, "steps": steps, "ambiguous": ambiguous, "calls": calls}
