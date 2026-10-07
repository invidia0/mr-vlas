"""Safety filters for velocity-controlled robots (book 12.4, 13.5): control barrier functions and a watchdog."""

import numpy as np


def cbf_filter(x, u_prop, zones=(), fence=None, others=(), sep=0.0, alpha=2.0, u_max=1.0, iters=3):
    """Minimally modify a proposed velocity so that a single integrator x' = u stays safe.

    zones: (centre, radius) keep-out circles; fence: (centre, radius) circle to stay inside, or a
    (low, high) axis-aligned box given as two arrays; others: positions of other robots to stay `sep` away
    from (each robot takes half of the avoidance effort). Constraints a.u >= b are enforced by sequential
    half-space projections (exact for one active constraint), then the speed limit scales the norm.
    Returns (u, active) where active is True when the command was modified.
    """
    x = np.asarray(x, float)
    cons = [(2 * (x - np.asarray(c)), -alpha * (np.sum((x - c) ** 2) - r ** 2)) for c, r in zones]
    if fence is not None:
        a0, a1 = (np.asarray(f, float) for f in fence)
        if a1.ndim == 0:                                          # circle: centre a0, radius a1
            cons.append((-2 * (x - a0), -alpha * (a1 ** 2 - np.sum((x - a0) ** 2))))
        else:                                                     # box: low a0, high a1, one constraint per face
            for k in range(len(x)):
                e = np.eye(len(x))[k]
                cons += [(e, -alpha * (x[k] - a0[k])), (-e, -alpha * (a1[k] - x[k]))]
    for y in others:
        d = x - np.asarray(y, float)
        cons.append((2 * d, -alpha / 2 * (d @ d - sep ** 2)))
    u = np.asarray(u_prop, float).copy()
    for _ in range(iters):
        for a, b in cons:
            gap = b - a @ u
            if gap > 0:
                u = u + gap / (a @ a + 1e-12) * a
    n = np.linalg.norm(u)
    if n > u_max:
        u = u * (u_max / n)
    return u, bool(np.linalg.norm(u - np.asarray(u_prop, float)) > 1e-9)


class Watchdog:
    """Replaces the command with a safe default when no fresh command arrived within `timeout` seconds;
    after `escalate` seconds of silence it latches a failsafe (e.g. land / return) until reset."""

    def __init__(self, timeout=0.3, escalate=2.0, safe=(0.0, 0.0)):
        self.timeout, self.escalate, self.safe = timeout, escalate, np.asarray(safe, float)
        self.last, self.failsafe = None, False

    def feed(self, t):
        self.last = t

    def __call__(self, t, u):
        silence = np.inf if self.last is None else t - self.last
        if silence > self.escalate:
            self.failsafe = True
        if self.failsafe or silence > self.timeout:
            return self.safe.copy(), True
        return np.asarray(u, float), False
