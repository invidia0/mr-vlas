"""The depot navigation task in MuJoCo physics (Lab 12; book Appendix A).

Same scenes, instructions, expert and interface as `vlacourse.depot_env.DepotNav`, but the robot is a 6 kg
cylinder on two actuated slide joints (a holonomic base) whose velocity actuators track the commanded velocity
with dynamics, and observations are rendered by an overhead MuJoCo camera.

Coordinates: the task's unit square [0, 1]^2 (x right, y down, as in the images) maps to a 4 m x 4 m floor:
world_x = 4 (x - 0.5), world_y = -4 (y - 0.5).
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from .data import scenes  # noqa: E402
from .depot_env import DepotNav  # noqa: E402

SIDE = 4.0          # metres covered by the unit square
CTRL_DT = 0.1       # control period (10 Hz)
SIM_DT = 0.005      # physics step (200 Hz)


def to_world(x, y):
    return SIDE * (x - 0.5), -SIDE * (y - 0.5)


def _object_xml(i, o):
    r, g, b = scenes.COLORS[o.color]
    rgba = f"{r} {g} {b} 1"
    dark = f"{0.55 * r} {0.55 * g} {0.55 * b} 1"
    wx, wy = to_world(o.x, o.y)
    if o.kind == "toolbox":
        geoms = (f'<geom type="box" size=".2 .12 .1" pos="0 0 .1" rgba="{rgba}"/>'
                 f'<geom type="box" size=".1 .015 .015" pos="0 0 .24" rgba=".1 .1 .1 1"/>'
                 f'<geom type="box" size=".015 .015 .03" pos=".09 0 .21" rgba=".1 .1 .1 1"/>'
                 f'<geom type="box" size=".015 .015 .03" pos="-.09 0 .21" rgba=".1 .1 .1 1"/>')
    elif o.kind == "crate":
        geoms = (f'<geom type="box" size=".18 .18 .18" pos="0 0 .18" rgba="{rgba}"/>'
                 + "".join(f'<geom type="box" size=".185 .02 .185" pos="0 {dy} .18" rgba="{dark}"/>' for dy in (-.09, .09)))
    elif o.kind == "barrel":
        geoms = (f'<geom type="cylinder" size=".17 .2" pos="0 0 .2" rgba="{rgba}"/>'
                 f'<geom type="cylinder" size=".1 .005" pos="0 0 .405" rgba="{dark}"/>')
    else:  # cone: stacked cylinders with a white band
        geoms = (f'<geom type="cylinder" size=".17 .05" pos="0 0 .05" rgba="{rgba}"/>'
                 f'<geom type="cylinder" size=".12 .06" pos="0 0 .16" rgba=".95 .95 .92 1"/>'
                 f'<geom type="cylinder" size=".07 .08" pos="0 0 .3" rgba="{rgba}"/>')
    # contype/conaffinity 0: the robot drives over objects' footprints instead of colliding (the task is reaching)
    return f'<body name="obj{i}" pos="{wx} {wy} 0">{geoms.replace("<geom ", "<geom contype=\"0\" conaffinity=\"0\" ")}</body>'


def depot_xml(scene, robot_xy=(0.5, 0.5)):
    wx, wy = to_world(*robot_xy)
    objects = "".join(_object_xml(i, o) for i, o in enumerate(scene.objects))
    return f"""
<mujoco model="depot">
  <option timestep="{SIM_DT}" integrator="implicitfast"/>
  <visual><headlight ambient=".45 .45 .45" diffuse=".25 .25 .25" specular="0 0 0"/><global offwidth="256" offheight="256"/></visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1=".72 .7 .66" rgb2=".67 .65 .61" width="256" height="256"/>
    <material name="floor" texture="grid" texrepeat="4 4"/>
  </asset>
  <worldbody>
    <light pos="0 0 6" dir="0 0 -1" directional="true" diffuse=".35 .35 .35" specular="0 0 0"/>
    <geom name="floor" type="plane" size="{SIDE / 2} {SIDE / 2} .1" material="floor"/>
    <camera name="top" pos="0 0 {SIDE / 2 / np.tan(np.radians(22.5)):.4f}" xyaxes="1 0 0 0 1 0" fovy="45"/>
    {objects}
    <body name="robot" pos="{wx} {wy} .08">
      <joint name="x" type="slide" axis="1 0 0" damping="2"/>
      <joint name="y" type="slide" axis="0 1 0" damping="2"/>
      <geom type="cylinder" size=".2 .08" mass="6" rgba=".12 .12 .14 1"/>
      <geom type="cylinder" size=".1 .085" rgba=".97 .97 .97 1" contype="0" conaffinity="0" mass="0"/>
      <camera name="onboard" pos="0 0 .3" xyaxes="0 -1 0 0 0 1" fovy="70"/>
    </body>
  </worldbody>
  <actuator>
    <velocity name="vx" joint="x" kv="60" ctrlrange="-1.6 1.6"/>
    <velocity name="vy" joint="y" kv="60" ctrlrange="-1.6 1.6"/>
  </actuator>
</mujoco>"""


class MujocoDepotNav(DepotNav):
    """DepotNav with MuJoCo dynamics and rendering. Actions are the same unit-disc velocity commands,
    scaled to at most v_step * SIDE / CTRL_DT = 1.6 m/s."""

    renderer_cache: dict = {}

    def _build(self):
        self.model = mujoco.MjModel.from_xml_string(depot_xml(self.scene, self.p))
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        key = (self.res,)
        old = self.renderer_cache.pop(key, None)
        if old is not None:
            old.close()
        self.renderer = mujoco.Renderer(self.model, self.res, self.res)
        self.renderer_cache[key] = self.renderer

    def reset(self, seed=None, keep_scene=False):
        self._ready = False
        super().reset(seed, keep_scene)              # chooses scene, instruction, targets, start p
        self._start = self.p.copy()
        self._build()
        self._ready = True
        return self.obs()

    def _position(self):
        wx, wy = self.data.qpos[0], self.data.qpos[1]
        base = np.array(to_world(*self._start))
        x = (base[0] + wx) / SIDE + 0.5
        y = -(base[1] + wy) / SIDE + 0.5
        return np.array([x, y])

    def obs(self):
        if not getattr(self, "_ready", False):      # called inside DepotNav.reset before the model exists
            return None
        self.renderer.update_scene(self.data, camera="top")
        img = self.renderer.render().copy()
        return {"image": img, "state": self.p.astype(np.float32).copy(), "instruction": self.instruction}

    def step(self, action):
        a = np.asarray(action, dtype=np.float64)
        n = np.linalg.norm(a)
        if n > 1:
            a = a / n
        speed = self.v_step * SIDE / CTRL_DT                       # m/s at full command
        self.data.ctrl[:] = [a[0] * speed, -a[1] * speed]          # image y points down, world y up
        for _ in range(int(round(CTRL_DT / SIM_DT))):
            mujoco.mj_step(self.model, self.data)
        self.p = np.clip(self._position(), 0.0, 1.0)
        self.t += 1
        success = self.distance_to_targets() < self.success_radius
        done = success or self.t >= self.max_steps
        return self.obs(), done, success

