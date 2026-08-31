#!/usr/bin/env python3
"""
waypoint_follower_p2p.py

One reusable node, run once per robot (with different parameters), that:
  1. Follows a fixed list of (x, y) waypoints using its own /odom.
  2. Subscribes DIRECTLY to the other robots' /odom topics - no server,
     no broker. This is the P2P layer: DDS discovery makes /amr2/odom and
     /amr3/odom visible to amr1's node automatically.
  3. Detects proximity/conflict with other robots and applies a simple
     priority-based yield rule (lower-priority robot slows/stops).
  4. Logs every conflict event to terminal + a log file, in a format that
     doubles as your "P2P info sharing" evidence for the report/demo, and
     as the data feed a frontend can later tail.

USAGE (run once per robot, three terminals/processes, one per namespace):

  ros2 run bcr_bot waypoint_follower_p2p.py --ros-args \
      -p self_namespace:=amr1 \
      -p other_namespaces:="['amr2','amr3']" \
      -p priority:=1 \
      -p waypoints_x:="[0.0, 3.0, 3.0, 0.0]" \
      -p waypoints_y:="[0.0, 0.0, 3.0, 3.0]"

Repeat for amr2 (priority:=2) and amr3 (priority:=3) with their own
waypoint lists. LOWER priority number wins right-of-way in a conflict
(i.e. priority 1 never yields to priority 2 or 3).

This file can be dropped into bcr_bot/scripts/ and added to the package's
console_scripts / install rules, OR just run directly with `python3` from
a sourced ROS2 shell for now - no need to rebuild the whole package to
iterate on this logic.
"""

import math
import time
from datetime import datetime

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist

LOG_FILE = "/tmp/p2p_conflict_log.txt"

# Tunable behaviour constants
CONFLICT_DISTANCE_M = 1.5     # start considering "in proximity"
YIELD_DISTANCE_M = 1.0        # actually slow/stop if this close AND lower priority
GOAL_TOLERANCE_M = 0.15
LINEAR_SPEED = 0.35
ANGULAR_SPEED = 0.8
YIELD_LINEAR_SCALE = 0.0      # 0.0 = full stop when yielding; try 0.2 for "slow" instead of "stop"


def yaw_from_quaternion(q):
    # Standard quaternion -> yaw (Z-axis rotation) conversion
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class WaypointFollowerP2P(Node):
    def __init__(self):
        super().__init__('waypoint_follower_p2p')

        # --- Parameters (set per-robot at launch) ---
        self.declare_parameter('self_namespace', 'amr1')
        self.declare_parameter('other_namespaces', ['amr2', 'amr3'])
        self.declare_parameter('priority', 1)  # lower = higher right-of-way
        self.declare_parameter('waypoints_x', [0.0, 3.0, 3.0, 0.0])
        self.declare_parameter('waypoints_y', [0.0, 0.0, 3.0, 3.0])

        self.self_ns = self.get_parameter('self_namespace').value
        self.other_ns_list = self.get_parameter('other_namespaces').value
        self.priority = self.get_parameter('priority').value
        wx = self.get_parameter('waypoints_x').value
        wy = self.get_parameter('waypoints_y').value
        self.waypoints = list(zip(wx, wy))
        self.wp_index = 0

        # --- State ---
        self.self_pose = None          # (x, y, yaw)
        self.other_poses = {}          # namespace -> (x, y, yaw, last_update_time)
        self.other_priorities = {}     # namespace -> priority (declared, since we know our own fleet config)
        # NOTE: in a real deployment each robot would broadcast its own priority
        # too (e.g. in a custom Intent message). For this first version we hardcode
        # the fleet's priority map so every robot can reason about right-of-way
        # without a central registry - this is still fully decentralized info,
        # just baked into each robot's own config rather than looked up centrally.
        self.fleet_priority = {'amr1': 1, 'amr2': 2, 'amr3': 3}

        qos = QoSProfile(depth=10,
                          reliability=ReliabilityPolicy.BEST_EFFORT,
                          history=HistoryPolicy.KEEP_LAST)

        # Subscribe to OWN odom
        self.create_subscription(
            Odometry, f'/{self.self_ns}/odom', self.self_odom_cb, qos)

        # Subscribe DIRECTLY to each other robot's odom - this is the P2P link.
        # No server in between: DDS discovery makes these topics visible the
        # moment the other robot's node comes up, in any order, on any host
        # on the same DDS domain.
        for ns in self.other_ns_list:
            self.create_subscription(
                Odometry, f'/{ns}/odom',
                lambda msg, ns=ns: self.other_odom_cb(msg, ns), qos)

        self.cmd_pub = self.create_publisher(Twist, f'/{self.self_ns}/cmd_vel', 10)

        self.timer = self.create_timer(0.1, self.control_loop)  # 10 Hz

        self.log_file = open(LOG_FILE, 'a')
        self._log(f"=== {self.self_ns} node started | priority={self.priority} | "
                   f"waypoints={self.waypoints} ===")

    # ------------------------------------------------------------------
    def _log(self, msg):
        line = f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] {msg}"
        print(line, flush=True)
        self.log_file.write(line + "\n")
        self.log_file.flush()

    # ------------------------------------------------------------------
    def self_odom_cb(self, msg: Odometry):
        p = msg.pose.pose.position
        yaw = yaw_from_quaternion(msg.pose.pose.orientation)
        self.self_pose = (p.x, p.y, yaw)

    def other_odom_cb(self, msg: Odometry, ns: str):
        p = msg.pose.pose.position
        yaw = yaw_from_quaternion(msg.pose.pose.orientation)
        self.other_poses[ns] = (p.x, p.y, yaw, time.time())

    # ------------------------------------------------------------------
    def current_goal(self):
        return self.waypoints[self.wp_index]

    def advance_waypoint(self):
        self.wp_index = (self.wp_index + 1) % len(self.waypoints)
        self._log(f"{self.self_ns}: reached waypoint, advancing to "
                   f"{self.waypoints[self.wp_index]}")

    # ------------------------------------------------------------------
    def check_conflicts(self):
        """
        Returns (should_yield: bool, nearest_conflict_ns: str or None,
                 nearest_dist: float or None)
        """
        if self.self_pose is None:
            return False, None, None

        sx, sy, _ = self.self_pose
        nearest_ns, nearest_dist = None, float('inf')

        for ns, (ox, oy, oyaw, t) in self.other_poses.items():
            # Ignore stale data (robot not publishing recently -
            # protects against acting on dropped/late packets)
            if time.time() - t > 1.0:
                continue
            dist = math.hypot(sx - ox, sy - oy)
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_ns = ns

        if nearest_ns is None or nearest_dist > CONFLICT_DISTANCE_M:
            return False, None, None

        other_priority = self.fleet_priority.get(nearest_ns, 999)

        # Log every proximity event once it's within "conflict" range,
        # regardless of who yields - this is the P2P sharing evidence.
        ox, oy, oyaw, t = self.other_poses[nearest_ns]
        self._log(
            f"[PROXIMITY] {self.self_ns} <-> {nearest_ns} | dist={nearest_dist:.2f}m | "
            f"{self.self_ns}: pos=({sx:.2f},{sy:.2f}) prio={self.priority} "
            f"next_wp={self.current_goal()} | "
            f"{nearest_ns}: pos=({ox:.2f},{oy:.2f}) prio={other_priority}"
        )

        should_yield = (
            nearest_dist < YIELD_DISTANCE_M and self.priority > other_priority
        )

        if should_yield:
            self._log(f"[YIELD] {self.self_ns} yielding to {nearest_ns} "
                      f"(dist={nearest_dist:.2f}m, {self.self_ns} prio={self.priority} > "
                      f"{nearest_ns} prio={other_priority})")

        return should_yield, nearest_ns, nearest_dist

    # ------------------------------------------------------------------
    def control_loop(self):
        if self.self_pose is None:
            return  # no odom yet

        sx, sy, syaw = self.self_pose
        gx, gy = self.current_goal()

        dist_to_goal = math.hypot(gx - sx, gy - sy)
        if dist_to_goal < GOAL_TOLERANCE_M:
            self.advance_waypoint()
            return

        # Simple proportional heading controller
        target_yaw = math.atan2(gy - sy, gx - sx)
        yaw_error = math.atan2(math.sin(target_yaw - syaw), math.cos(target_yaw - syaw))

        should_yield, conflict_ns, conflict_dist = self.check_conflicts()

        cmd = Twist()
        if should_yield:
            cmd.linear.x = LINEAR_SPEED * YIELD_LINEAR_SCALE
            cmd.angular.z = 0.0
        else:
            # Turn first if heading is far off, otherwise drive forward
            if abs(yaw_error) > 0.3:
                cmd.linear.x = 0.0
                cmd.angular.z = ANGULAR_SPEED if yaw_error > 0 else -ANGULAR_SPEED
            else:
                cmd.linear.x = LINEAR_SPEED
                cmd.angular.z = 0.5 * yaw_error

        self.cmd_pub.publish(cmd)

    def destroy_node(self):
        self.log_file.close()
        super().destroy_node()


def main():
    rclpy.init()
    node = WaypointFollowerP2P()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()