"""Offboard node with a safety supervisor between VLA goals and PX4 (book 13.2, 13.3, 13.5; Appendix B).

Subscribes:  /vla/goal (geometry_msgs/PointStamped, local ENU metres)  — proposals from a VLA / VLM / script
             PX4 vehicle local position (px4_msgs/VehicleLocalPosition, NED)
Publishes:   /fmu/in/offboard_control_mode, /fmu/in/trajectory_setpoint (velocity, NED), /fmu/in/vehicle_command
             /vla/supervisor_status (std_msgs/String) — every decision of the supervisor, for logging

Layers (all on the companion): goal validation -> straight-line plan with speed/acceleration limits ->
CBF filter for no-fly zones and the fence -> watchdog (hold when goals stop). PX4's own geofence and failsafes
stay configured independently.
"""

import math

import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand, VehicleLocalPosition
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

from .safety import cbf_filter

QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                 history=HistoryPolicy.KEEP_LAST, depth=1)


def ned_to_enu(v):
    return np.array([v[1], v[0], -v[2]])


def enu_to_ned(v):
    return np.array([v[1], v[0], -v[2]])


class OffboardSupervisor(Node):
    def __init__(self):
        super().__init__("offboard_supervisor")
        p = self.declare_parameters("", [
            ("local_position_topic", "/fmu/out/vehicle_local_position"),   # may carry a version suffix
            ("rate_hz", 20.0), ("v_max", 2.0), ("a_max", 1.5), ("z_min", 2.0), ("z_takeoff", 5.0),
            ("fence_low", [-30.0, -30.0, 0.0]), ("fence_high", [30.0, 30.0, 20.0]),
            ("no_fly", [10.0, 0.0, 4.0]),             # flat list: x, y, radius, x, y, radius, ...
            ("margin", 1.0), ("alpha", 1.0), ("goal_timeout_s", 3.0), ("k_pos", 0.8)])
        g = {x.name: x.value for x in p}
        self.g = g
        self.low, self.high = np.array(g["fence_low"]), np.array(g["fence_high"])
        nf = g["no_fly"]
        self.zones = [(np.array(nf[i:i + 2]), nf[i + 2]) for i in range(0, len(nf), 3)]
        self.pos = None
        self.goal, self.goal_t = None, None
        self.ticks = 0
        self.mode_pub = self.create_publisher(OffboardControlMode, "/fmu/in/offboard_control_mode", QOS)
        self.sp_pub = self.create_publisher(TrajectorySetpoint, "/fmu/in/trajectory_setpoint", QOS)
        self.cmd_pub = self.create_publisher(VehicleCommand, "/fmu/in/vehicle_command", QOS)
        self.status_pub = self.create_publisher(String, "/vla/supervisor_status", 10)
        self.create_subscription(VehicleLocalPosition, g["local_position_topic"], self.on_position, QOS)
        self.create_subscription(PointStamped, "/vla/goal", self.on_goal, 10)
        self.v_cmd = np.zeros(3)
        self.create_timer(1.0 / g["rate_hz"], self.tick)

    # ------------------------------------------------------------------ inputs
    def now_s(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def status(self, text):
        self.status_pub.publish(String(data=text))
        self.get_logger().info(text)

    def on_position(self, msg):
        if msg.xy_valid and msg.z_valid:
            self.pos = ned_to_enu([msg.x, msg.y, msg.z])

    def on_goal(self, msg):
        g = np.array([msg.point.x, msg.point.y, msg.point.z])
        reason = self.validate(g)
        if reason:
            self.status(f"goal {np.round(g, 1).tolist()} REJECTED: {reason}")
            return
        self.goal, self.goal_t = g, self.now_s()
        self.status(f"goal {np.round(g, 1).tolist()} accepted")

    def validate(self, g):
        if (g < self.low).any() or (g > self.high).any():
            return "outside the operational volume"
        if g[2] < self.g["z_min"]:
            return "below minimum altitude"
        for c, r in self.zones:
            if np.linalg.norm(g[:2] - c) < r + self.g["margin"]:
                return "inside a no-fly zone"
        return None

    # ------------------------------------------------------------------ control loop
    def tick(self):
        self.mode_pub.publish(OffboardControlMode(timestamp=self.us(), velocity=True))
        if self.ticks == int(self.g["rate_hz"]):                       # after 1 s of streaming
            self.command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
            self.command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
            self.status("requested offboard + arm")
        self.ticks += 1
        v = np.zeros(3)
        if self.pos is not None:
            target = self.goal
            stale = self.goal_t is None or self.now_s() - self.goal_t > self.g["goal_timeout_s"]
            if target is None:                                          # take off and hold until a goal arrives
                target = np.array([self.pos[0], self.pos[1], self.g["z_takeoff"]])
            v = self.plan(target)
            if stale and self.goal is not None:
                v = np.zeros(3)                                         # watchdog: hold position
                if self.ticks % int(self.g["rate_hz"]) == 0:
                    self.status("watchdog: no fresh goal, holding")
            v_xy, active = cbf_filter(self.pos[:2], v[:2], [(c, r + self.g["margin"]) for c, r in self.zones],
                                      fence=(self.low[:2] + self.g["margin"], self.high[:2] - self.g["margin"]),
                                      alpha=self.g["alpha"], u_max=self.g["v_max"])
            if active and self.ticks % int(self.g["rate_hz"]) == 0:
                self.status("CBF filter active")
            v = np.array([v_xy[0], v_xy[1], float(np.clip(v[2], -1.0, 1.0))])
        self.v_cmd = v
        sp = TrajectorySetpoint(timestamp=self.us())
        sp.position = [math.nan] * 3
        sp.velocity = enu_to_ned(v).astype(np.float32).tolist()
        sp.yaw = math.nan
        self.sp_pub.publish(sp)

    def plan(self, target):
        """Velocity towards the target with a braking-distance speed limit (a straight-line planner)."""
        d = target - self.pos
        dist = float(np.linalg.norm(d))
        if dist < 0.2:
            return np.zeros(3)
        speed = min(self.g["v_max"], self.g["k_pos"] * dist, math.sqrt(2 * self.g["a_max"] * dist))
        return d / dist * speed

    # ------------------------------------------------------------------ PX4 helpers
    def us(self):
        return int(self.get_clock().now().nanoseconds / 1000)

    def command(self, cmd, p1=0.0, p2=0.0):
        self.cmd_pub.publish(VehicleCommand(timestamp=self.us(), command=cmd, param1=p1, param2=p2, target_system=1,
                                            target_component=1, source_system=1, source_component=1,
                                            from_external=True))


def main():
    rclpy.init()
    rclpy.spin(OffboardSupervisor())


if __name__ == "__main__":
    main()
