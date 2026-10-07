"""A stand-in for the VLA: publishes goals on /vla/goal and injects failures (Appendix B, book 13.6).

    ros2 run vla_px4_bridge goal_source --ros-args -p scenario:=nominal
    scenarios: nominal | no_fly_goal | outside_fence | silence | through_zone

Replace this node by one that runs a VLA/VLM on the camera stream and publishes the grounded goal (ENU, metres).
"""

import rclpy
from geometry_msgs.msg import PointStamped
from rclpy.node import Node

SCENARIOS = {
    "nominal": [(5.0, 5.0, 5.0), (5.0, 15.0, 6.0), (-10.0, 10.0, 5.0), (0.0, 0.0, 5.0)],   # away from the zone
    "no_fly_goal": [(5.0, 0.0, 5.0), (10.0, 0.0, 5.0), (0.0, 0.0, 5.0)],        # the 2nd is inside the zone
    "outside_fence": [(5.0, 5.0, 5.0), (60.0, 0.0, 5.0)],                        # the 2nd must be rejected
    "silence": [(5.0, 5.0, 5.0)],                                               # then stop: watchdog must hold
    "through_zone": [(0.0, 1.5, 5.0), (20.0, 1.5, 5.0)],                          # path crosses the zone: CBF
}


class GoalSource(Node):
    def __init__(self):
        super().__init__("goal_source")
        self.declare_parameter("scenario", "nominal")
        self.declare_parameter("period_s", 8.0)
        self.goals = SCENARIOS[self.get_parameter("scenario").value]
        self.k = 0
        self.pub = self.create_publisher(PointStamped, "/vla/goal", 10)
        self.create_timer(1.0, self.republish)                                    # proposals are refreshed at 1 Hz
        self.create_timer(self.get_parameter("period_s").value, self.next_goal)

    def next_goal(self):
        self.k += 1

    def republish(self):
        silent = self.get_parameter("scenario").value == "silence" and self.k >= 1
        if silent or self.k >= len(self.goals):
            return
        msg = PointStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "local_enu"
        msg.point.x, msg.point.y, msg.point.z = self.goals[self.k]
        self.pub.publish(msg)


def main():
    rclpy.init()
    rclpy.spin(GoalSource())


if __name__ == "__main__":
    main()
