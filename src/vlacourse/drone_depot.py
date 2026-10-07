"""The depot seen from a drone (Lab 15; book Chapter 13).

Same scenes, instructions, targets and top-down images as `DepotNav`, but the vehicle tracks a *velocity
setpoint* through first-order dynamics (time constant `tau`) under wind gusts (an Ornstein-Uhlenbeck process),
integrated at 50 Hz while decisions arrive at 10 Hz. Keep-out zones and a geofence are part of the map, not of
the image: the policy never sees them, the safety layer does.
"""

from dataclasses import dataclass, field

import numpy as np

from .depot_env import DepotNav


@dataclass
class DroneDepot(DepotNav):
    max_steps: int = 80
    tau: float = 0.25            # velocity time constant (s)
    wind: float = 0.06           # gust strength (units/s)
    substeps: int = 5            # 50 Hz physics under 10 Hz control
    dt: float = 0.1              # control period (s)
    zone_prob: float = 0.0       # probability that a keep-out zone sits across the path to the target
    zone_radius: float = 0.08
    fence: tuple = ((0.03, 0.03), (0.97, 0.97))
    zones: list = field(default_factory=list)
    hold_steps: int = 1          # success = within success_radius for this many consecutive control steps

    @property
    def speed(self):             # units/s at full command (matches DepotNav: v_step per 0.1 s)
        return self.v_step / self.dt

    def reset(self, seed=None, keep_scene=False):
        obs = super().reset(seed, keep_scene)
        rng = np.random.default_rng(self.seed)
        self.v, self.w = np.zeros(2), np.zeros(2)
        self.wrng = rng
        self.zones = []
        if rng.random() < self.zone_prob:                      # across the path, slightly off-axis
            g = np.array([self.goal.x, self.goal.y])
            d = g - self.p
            n = np.array([-d[1], d[0]]) / (np.linalg.norm(d) + 1e-9)
            c = self.p + 0.5 * d + n * rng.choice([-1, 1]) * rng.uniform(0.02, 0.04)
            self.zones = [(c, self.zone_radius)]
        self.min_clearance = self.clearance()
        self.fence_exit = False
        self.inside = 0
        return obs

    def clearance(self):
        """Signed distance to the nearest keep-out zone (inf without zones); negative = inside."""
        return min((np.linalg.norm(self.p - c) - r for c, r in self.zones), default=np.inf)

    def step(self, command):
        """command: velocity setpoint in the unit disc (scaled by `speed`)."""
        u = np.asarray(command, float)
        n = np.linalg.norm(u)
        if n > 1:
            u = u / n
        h = self.dt / self.substeps
        for _ in range(self.substeps):
            self.w += -self.w * h / 0.5 + self.wind * np.sqrt(h) * self.wrng.normal(size=2)
            self.v += (u * self.speed - self.v) * h / self.tau
            self.p = self.p + (self.v + self.w) * h
            self.min_clearance = min(self.min_clearance, self.clearance())
            lo, hi = np.array(self.fence[0]), np.array(self.fence[1])
            self.fence_exit |= bool((self.p < lo).any() or (self.p > hi).any())
        self.p = np.clip(self.p, 0.0, 1.0)
        self.t += 1
        self.inside = self.inside + 1 if self.distance_to_targets() < self.success_radius else 0
        success = self.inside >= self.hold_steps
        done = success or self.t >= self.max_steps
        return self.obs(), done, success
