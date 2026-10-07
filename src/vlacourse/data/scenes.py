"""Procedural top-down depot scenes: images, captions and question-answer pairs.

A scene is a few objects (toolbox, crate, barrel, cone in four colours) lying on
the depot floor, seen from above. The same scenes are used to train the toy
vision encoder (Lab 06), the toy CLIP (Lab 07) and the toy VLM (Lab 08), and
later the toy VLA, so that every model in the book sees the same world.

Coordinates: x to the right, y downwards (image rows), both in [0, 1].
"""

import random
import re
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

KINDS = ["toolbox", "crate", "barrel", "cone"]
COLORS = {
    "red": (0.84, 0.16, 0.14),
    "blue": (0.18, 0.38, 0.86),
    "green": (0.13, 0.62, 0.25),
    "yellow": (0.96, 0.80, 0.10),
}
FLOOR = (0.70, 0.68, 0.64)
NUMBERS = ["zero", "one", "two", "three", "four"]


@dataclass(frozen=True)
class Obj:
    kind: str
    color: str
    x: float  # centre, in [0, 1]
    y: float


@dataclass(frozen=True)
class Scene:
    objects: tuple[Obj, ...]


def random_scene(rng: random.Random, n_objects=(1, 3), min_dist: float = 0.3, margin: float = 0.15,
                 kinds=KINDS) -> Scene:
    """Non-overlapping objects with distinct (kind, colour) pairs, kinds drawn from `kinds`."""
    n = rng.randint(*n_objects) if isinstance(n_objects, tuple) else n_objects
    pairs = rng.sample([(k, c) for k in kinds for c in COLORS], n)
    objs: list[Obj] = []
    while len(objs) < n:
        x, y = rng.uniform(margin, 1 - margin), rng.uniform(margin, 1 - margin)
        if all((x - o.x) ** 2 + (y - o.y) ** 2 >= min_dist ** 2 for o in objs):
            k, c = pairs[len(objs)]
            objs.append(Obj(k, c, x, y))
    return Scene(tuple(objs))


# ------------------------------------------------------------------ rendering

def _masks(kind: str, u: np.ndarray, v: np.ndarray):
    """(body, detail) boolean masks in object coordinates u, v (object size = 1)."""
    if kind == "toolbox":
        body = (np.abs(u) < 0.5) & (v > -0.25) & (v < 0.35)
        handle = ((v > -0.45) & (v < -0.36) & (np.abs(u) < 0.25)) | \
                 ((v > -0.45) & (v < -0.25) & (np.abs(np.abs(u) - 0.22) < 0.04))
        stripe = body & (np.abs(v - 0.02) < 0.04)
        return body | handle, handle | stripe
    if kind == "crate":
        body = (np.abs(u) < 0.42) & (np.abs(v) < 0.42)
        slats = body & ((np.abs(v + 0.21) < 0.035) | (np.abs(v) < 0.035) | (np.abs(v - 0.21) < 0.035))
        return body, slats
    if kind == "barrel":
        r = np.hypot(u, v)
        return r < 0.42, (np.abs(r - 0.27) < 0.035) | (r < 0.06)
    if kind == "cone":
        body = (v > -0.45) & (v < 0.42) & (np.abs(u) < 0.48 * (v + 0.45) / 0.87)
        band = body & (np.abs(v - 0.05) < 0.07)
        return body, band
    raise ValueError(kind)


@lru_cache(maxsize=8)
def _floor(n: int) -> np.ndarray:
    yy, xx = (np.mgrid[0:n, 0:n] + 0.5) / n
    img = np.empty((n, n, 3))
    img[:] = FLOOR
    # concrete slabs: faint joints every quarter of the image
    joints = (np.abs((xx * 4) % 1 - 0.5) > 0.49) | (np.abs((yy * 4) % 1 - 0.5) > 0.49)
    img[joints] *= 0.93
    return img


def render(scene: Scene, res: int = 64, size: float = 0.22, texture: float = 0.03, seed: int = 0,
           supersample: int = 3, agents=(), agent_radius: float = 0.05) -> np.ndarray:
    """Render to a uint8 RGB image of shape (res, res, 3). `size` is the object size as a fraction of the image.

    agents: robots seen from above, drawn on top as dark discs with a coloured centre; each entry is
    (x, y) or (x, y, rgb).
    """
    n = res * supersample
    img = _floor(n).copy()
    half = int(np.ceil(0.5 * size * n)) + 1
    for o in scene.objects:
        # draw only inside the object's bounding box
        r0, c0 = max(0, int(o.y * n) - half), max(0, int(o.x * n) - half)
        r1, c1 = min(n, int(o.y * n) + half), min(n, int(o.x * n) + half)
        yy, xx = (np.mgrid[r0:r1, c0:c1] + 0.5) / n
        body, detail = _masks(o.kind, (xx - o.x) / size, (yy - o.y) / size)
        col = np.array(COLORS[o.color])
        patch = img[r0:r1, c0:c1]
        patch[body] = col
        patch[detail] = col * 0.55 if o.kind != "cone" else (0.95, 0.95, 0.92)
    for a in agents:
        x, y = a[0], a[1]
        centre = a[2] if len(a) > 2 else (0.97, 0.97, 0.97)
        h = int(np.ceil(agent_radius * n)) + 1
        r0, c0 = max(0, int(y * n) - h), max(0, int(x * n) - h)
        r1, c1 = min(n, int(y * n) + h), min(n, int(x * n) + h)
        yy, xx = (np.mgrid[r0:r1, c0:c1] + 0.5) / n
        r = np.hypot(xx - x, yy - y) / agent_radius
        patch = img[r0:r1, c0:c1]
        patch[r < 1] = (0.12, 0.12, 0.14)
        patch[r < 0.5] = centre
    img = img.reshape(res, supersample, res, supersample, 3).mean((1, 3))  # anti-aliasing
    if texture:
        img += np.random.default_rng(seed).normal(0, texture, img.shape)
    return (np.clip(img, 0, 1) * 255).round().astype(np.uint8)


# ------------------------------------------------------------------ language

def _name(o: Obj) -> str:
    return f"{o.color} {o.kind}"


def relation(a: Obj, b: Obj) -> str:
    """Spatial relation of a with respect to b, in image terms."""
    dx, dy = a.x - b.x, a.y - b.y
    if abs(dx) >= abs(dy):
        return "left of" if dx < 0 else "right of"
    return "above" if dy < 0 else "below"


def region(o: Obj) -> str:
    row = ["top", "middle", "bottom"][min(2, int(o.y * 3))]
    col = ["left", "center", "right"][min(2, int(o.x * 3))]
    return "center" if (row, col) == ("middle", "center") else f"{row} {col}"


def caption(scene: Scene) -> str:
    objs = sorted(scene.objects, key=lambda o: o.x)
    if len(objs) == 1:
        return f"a {_name(objs[0])}"
    if len(objs) == 2:
        return f"a {_name(objs[0])} {relation(objs[0], objs[1])} a {_name(objs[1])}"
    return ", ".join(f"a {_name(o)}" for o in objs[:-1]) + f" and a {_name(objs[-1])}"


def qa_pairs(scene: Scene) -> list[tuple[str, str]]:
    """All question-answer pairs that the scene determines unambiguously."""
    objs = scene.objects
    qa = [("how many objects are there?", NUMBERS[len(objs)])]
    kinds = [o.kind for o in objs]
    colors = [o.color for o in objs]
    for o in objs:
        if kinds.count(o.kind) == 1:
            qa.append((f"what color is the {o.kind}?", o.color))
        if colors.count(o.color) == 1:
            qa.append((f"what is the {o.color} object?", o.kind))
        qa.append((f"where is the {_name(o)}?", region(o)))
    if len(objs) == 2:
        a, b = objs
        qa.append((f"what is {relation(a, b)} the {_name(b)}?", a.kind))
        qa.append((f"what is {relation(b, a)} the {_name(a)}?", b.kind))
    return qa


# ------------------------------------------------------------------ word-level tokenizer

SPECIALS = ["<pad>", "<bos>", "<eos>", "<sep>"]
_WORDS = sorted(set(
    " ".join(KINDS + list(COLORS) + NUMBERS).split()
    + "a an and the of left right above below top middle bottom center how many objects are there what color "
      "is where object , ?".split()
))
VOCAB = SPECIALS + _WORDS
PAD, BOS, EOS, SEP = range(4)
_INDEX = {w: i for i, w in enumerate(VOCAB)}


def encode(text: str) -> list[int]:
    """Word-level ids (no specials). Raises KeyError on words outside the scene vocabulary."""
    return [_INDEX[w] for w in re.findall(r"[a-z]+|[,?]", text.lower())]


def decode(ids) -> str:
    words = [VOCAB[int(i)] for i in ids if int(i) not in (PAD, BOS)]
    if "<eos>" in words:
        words = words[:words.index("<eos>")]
    return " ".join(words).replace(" ,", ",").replace(" ?", "?")


def scene_dataset(n: int, seed: int = 0, res: int = 64, **kw):
    """n random scenes and their renders: (scenes, images uint8 (n, res, res, 3))."""
    rng = random.Random(seed)
    scenes = [random_scene(rng, **kw) for _ in range(n)]
    images = np.stack([render(s, res, seed=seed * 1_000_003 + i) for i, s in enumerate(scenes)])
    return scenes, images
