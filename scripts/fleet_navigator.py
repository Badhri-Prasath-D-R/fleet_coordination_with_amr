#!/usr/bin/env python3
"""Route planning, path following and rerouting for one AMR.

This is the layer that lets the fleet get *unstuck*. A speed governor can
only ever answer "slow down", and slowing down does not resolve a head-on
encounter in a 1.8 m aisle -- it only decides who reaches the stalemate
first. The way out is a different route, which is why the peers appear here
as well, in the planner cost function rather than in a brake.

Pipeline::

    task allocator --> nav_goal --> [ THIS NODE ] --> cmd_vel_nav
                                          |                |
                                          |                v
                                          |       fleet_collision_node
                                          |                |
                                          +<-- conflict_report

Peers enter planning two ways:

* **soft**, always: every known peer inflates the cost of the cells around
  it, so routes bend away from congestion before it becomes a conflict.
  This is the cheapest collision avoidance available -- the kind that
  happens thirty seconds early, on a route that never puts the two robots
  in the same aisle.
* **hard**, on escalation: when the collision layer reports that it has been
  held behind a specific peer, that peer becomes impassable for a while and
  the search has to find a genuinely different corridor, or admit there
  is not one.

When there is not one, the node backs off far enough to let the other robot
through, and if that still fails it hands the task back to the fleet. That
last step is what stops a single blocked aisle from stalling a job forever.
"""

import json
import math
import os

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Bool, String

from fleet_coordination.bus import (FLEET_STATE_TOPIC, PeerTable, encode,
                                    state_qos)
from fleet_coordination.control import PathFollower
from fleet_coordination.flog import FleetLogger
from fleet_coordination.geometry import yaw_from_quat
from fleet_coordination.gridmap import load_map
from fleet_coordination.planner import DynamicObstacle, Planner

IDLE = "IDLE"
PLANNING = "PLANNING"
FOLLOWING = "FOLLOWING"
BACKOFF = "BACKOFF"
WAITING = "WAITING"


class FleetNavigator(Node):

    def __init__(self):
        super().__init__("fleet_navigator")

        self.declare_parameters("", [
            ("robot_id", "amr1"),
            ("map_yaml", ""),
            ("planning_downsample", 3),

            # ---- chassis and clearance ----
            ("footprint_length", 0.90),
            ("footprint_width", 0.64),
            # Default is the circumscribed radius (0.45 m for this chassis),
            # not the half-width, so that every cell on a planned path is one
            # the robot could also turn around in.
            ("planning_clearance", 0.45),
            ("prefer_clearance", 0.90),
            ("clearance_weight", 0.6),

            # ---- velocity profile ----
            ("max_linear", 0.40),
            ("max_angular", 1.00),
            ("max_accel", 0.35),
            ("lookahead_m", 0.80),
            ("lookahead_gain", 0.9),
            ("lookahead_min", 0.45),
            ("lookahead_max", 1.60),
            ("spin_threshold_deg", 35.0),
            ("spin_gain", 1.6),
            ("goal_tolerance", 0.25),
            ("final_approach_m", 1.0),

            # ---- replanning ----
            ("replan_period_s", 4.0),
            ("peer_soft_radius", 1.40),
            ("peer_soft_weight", 4.0),
            ("peer_hard_radius", 0.85),
            ("blocker_memory_s", 12.0),
            ("stuck_peer_s", 4.0),
            ("max_reroutes", 3),
            ("max_backoffs", 3),
            ("wait_timeout_s", 12.0),

            # ---- retreat ----
            ("backoff_speed", 0.18),
            ("backoff_distance", 1.20),
            ("backoff_timeout_s", 10.0),

            # ---- plumbing ----
            ("control_hz", 20.0),
            ("peer_timeout_s", 1.5),
            ("path_frame", "map"),
            ("log_file", ""),
        ])

        g = self.get_parameter
        self.rid = g("robot_id").value
        self.width = float(g("footprint_width").value)
        self.length = float(g("footprint_length").value)

        self.max_v = float(g("max_linear").value)
        self.max_w = float(g("max_angular").value)
        # The follower is the same object the offline simulator drives, so
        # what is validated in scripts/simulate_fleet.py is this code, not a
        # second implementation of it.
        self.follower = PathFollower(
            max_linear=self.max_v,
            max_angular=self.max_w,
            max_accel=float(g("max_accel").value),
            lookahead_m=float(g("lookahead_m").value),
            lookahead_gain=float(g("lookahead_gain").value),
            lookahead_min=float(g("lookahead_min").value),
            lookahead_max=float(g("lookahead_max").value),
            spin_threshold_rad=math.radians(float(g("spin_threshold_deg").value)),
            spin_gain=float(g("spin_gain").value),
            goal_tolerance=float(g("goal_tolerance").value),
            final_approach_m=float(g("final_approach_m").value))
        self.goal_tol = float(g("goal_tolerance").value)

        self.replan_period = float(g("replan_period_s").value)
        self.soft_radius = float(g("peer_soft_radius").value)
        self.soft_weight = float(g("peer_soft_weight").value)
        self.hard_radius = float(g("peer_hard_radius").value)
        self.blocker_memory = float(g("blocker_memory_s").value)
        self.stuck_peer_s = float(g("stuck_peer_s").value)
        self.max_reroutes = int(g("max_reroutes").value)
        self.max_backoffs = int(g("max_backoffs").value)
        self.wait_timeout = float(g("wait_timeout_s").value)

        self.backoff_speed = float(g("backoff_speed").value)
        self.backoff_distance = float(g("backoff_distance").value)
        self.backoff_timeout = float(g("backoff_timeout_s").value)

        self.dt = 1.0 / max(1.0, float(g("control_hz").value))
        self.path_frame = g("path_frame").value

        self.log = FleetLogger(self.rid, path=(g("log_file").value or None),
                               sim_clock=self.now, ros_logger=self.get_logger())

        # ---- map and planner ----
        map_yaml = g("map_yaml").value
        self.grid = None
        self.planner = None
        if map_yaml and os.path.exists(os.path.expanduser(map_yaml)):
            self.grid = load_map(os.path.expanduser(map_yaml),
                                 downsample=int(g("planning_downsample").value))
            self.planner = Planner(
                self.grid,
                min_clearance=float(g("planning_clearance").value),
                prefer_clearance=float(g("prefer_clearance").value),
                clearance_weight=float(g("clearance_weight").value))
            self.get_logger().info("[%s] map %s -> %s"
                                   % (self.rid, map_yaml, self.grid))
        else:
            # Without a map the node still works, it just cannot route around
            # anything: goals are pursued in a straight line. Saying so is
            # better than pretending the reroute layer is armed when it is not.
            self.get_logger().warn(
                "[%s] no map at %r -- straight-line mode, rerouting disabled"
                % (self.rid, map_yaml))

        # ---- state ----
        self.pose = None
        self.speed = 0.0
        self.peers = PeerTable(timeout_s=float(g("peer_timeout_s").value))
        self.peer_still_since = {}

        self.state = IDLE
        self.goal = None
        self.task_id = None
        self.phase = "IDLE"
        self.path = []
        self.follower.reset()
        self.cmd_v = 0.0
        self.hard_blockers = {}          # rid -> expiry time
        self.reroute_count = 0
        self.backoffs = 0
        self.last_plan_t = -1e9
        self.state_since = 0.0
        self.backoff_start = None
        self.backoff_reason = ""
        self.pending_reroute = False

        # ---- ROS interfaces ----
        qos = state_qos()
        self.create_subscription(Odometry, "odom", self.on_odom, qos)
        self.create_subscription(String, FLEET_STATE_TOPIC, self.on_peer, qos)
        self.create_subscription(String, "nav_goal", self.on_nav_goal, 10)
        self.create_subscription(PoseStamped, "goal_pose", self.on_goal_pose, 10)
        self.create_subscription(String, "conflict_report", self.on_conflict, 10)
        self.create_subscription(Bool, "reroute_request", self.on_reroute_bool, 10)

        self.pub_cmd = self.create_publisher(Twist, "cmd_vel_nav", 10)
        self.pub_path = self.create_publisher(Path, "plan", 5)
        self.pub_intent = self.create_publisher(String, "intent", 10)
        self.pub_result = self.create_publisher(String, "nav_result", 10)

        self.create_timer(self.dt, self.tick)
        self.create_timer(0.2, self.publish_intent)

        self.get_logger().info(
            "[%s] navigator up | v_max=%.2f w_max=%.2f clearance=%.2fm"
            % (self.rid, self.max_v, self.max_w,
               float(g("planning_clearance").value)))

    # ------------------------------------------------------------- clock

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    # --------------------------------------------------------------- I/O

    def on_odom(self, msg):
        p = msg.pose.pose.position
        self.pose = (p.x, p.y, yaw_from_quat(msg.pose.pose.orientation))
        self.speed = abs(msg.twist.twist.linear.x)

    def on_peer(self, msg):
        try:
            payload = json.loads(msg.data)
        except (ValueError, TypeError):
            return
        if payload.get("id") == self.rid:
            return
        now = self.now()
        st = self.peers.update(payload, now)
        if st is None:
            return
        if st.speed < 0.05:
            self.peer_still_since.setdefault(st.rid, now)
        else:
            self.peer_still_since.pop(st.rid, None)

    def on_nav_goal(self, msg):
        """A goal from the task allocator."""
        try:
            d = json.loads(msg.data)
        except (ValueError, TypeError):
            return
        if d.get("cancel"):
            self.abandon("cancelled")
            return
        self.accept_goal((float(d["x"]), float(d["y"])),
                         task_id=d.get("task"), phase=d.get("phase", "TO_GOAL"))

    def on_goal_pose(self, msg):
        """A goal poked in by hand, e.g. from RViz or ros2 topic pub."""
        self.accept_goal((msg.pose.position.x, msg.pose.position.y),
                         task_id=None, phase="MANUAL")

    def on_conflict(self, msg):
        """The collision layer has been held behind a specific peer.

        This is the trigger that promotes a peer from soft cost to hard
        obstacle. It carries the peer id and the reason, which is more than
        the bare Bool can express -- head-on and "it is parked in my way"
        deserve the same escalation, but a generic deadlock does not need to
        wall off the peer for as long.
        """
        try:
            d = json.loads(msg.data)
        except (ValueError, TypeError):
            return
        peer = d.get("peer")
        if peer:
            self.hard_blockers[peer] = self.now() + self.blocker_memory
        self.pending_reroute = True
        self.log.reroute(d.get("reason", "conflict"), peer_id=peer,
                         attempt=self.reroute_count + 1)

    def on_reroute_bool(self, msg):
        if msg.data:
            self.pending_reroute = True

    # ------------------------------------------------------------- goals

    def accept_goal(self, goal, task_id, phase):
        self.goal = goal
        self.task_id = task_id
        self.phase = phase
        self.reroute_count = 0
        self.backoffs = 0
        self.hard_blockers.clear()
        self.pending_reroute = False
        self.enter(PLANNING)
        self.log.task("goal", task_id or "manual",
                      "phase=%s target=(%.2f,%.2f)" % (phase, goal[0], goal[1]))

    def abandon(self, reason):
        self.goal = None
        self.path = []
        self.cmd_v = 0.0
        self.enter(IDLE)
        self.phase = "IDLE"
        self.log.task("abandon", self.task_id or "manual", "reason=%s" % reason)
        self.task_id = None

    def enter(self, state):
        if state != self.state:
            self.get_logger().info("[%s] nav %s -> %s"
                                   % (self.rid, self.state, state))
        self.state = state
        self.state_since = self.now()

    def report(self, result, detail=""):
        self.pub_result.publish(encode({
            "id": self.rid, "task": self.task_id, "phase": self.phase,
            "result": result, "detail": detail, "t": round(self.now(), 3)}))

    # -------------------------------------------------------- obstacles

    def dynamic_layer(self):
        """Peers, as the planner should see them right now."""
        now = self.now()
        for rid in [r for r, exp in self.hard_blockers.items() if exp < now]:
            del self.hard_blockers[rid]

        out = []
        for rid, peer in self.peers.items():
            stuck_since = self.peer_still_since.get(rid)
            long_stuck = (stuck_since is not None
                          and now - stuck_since > self.stuck_peer_s)
            hard = rid in self.hard_blockers or long_stuck
            out.append(DynamicObstacle(
                peer.x, peer.y,
                self.hard_radius if hard else self.soft_radius,
                hard=hard, weight=self.soft_weight, rid=rid))
        return out

    # ----------------------------------------------------------- planning

    def replan(self, reason):
        """Plan to the current goal. Returns True on success."""
        if self.goal is None or self.pose is None:
            return False
        self.last_plan_t = self.now()

        if self.planner is None:                    # no map: fly straight
            self.path = [(self.pose[0], self.pose[1]), self.goal]
            self.follower.reset()
            self.publish_path()
            return True

        dynamic = self.dynamic_layer()
        old_len = Planner.path_length(self.path) if self.path else None

        path = self.planner.plan((self.pose[0], self.pose[1]), self.goal,
                                 dynamic=dynamic)
        if path is None and any(d.hard for d in dynamic):
            # Nothing gets through with the blockers treated as walls. Try
            # again with them soft: a long detour that merely passes near
            # the blocker is still better than giving up, and the collision
            # layer is underneath us either way.
            soft = [DynamicObstacle(d.x, d.y, self.soft_radius, hard=False,
                                    weight=self.soft_weight * 2.0, rid=d.rid)
                    for d in dynamic]
            path = self.planner.plan((self.pose[0], self.pose[1]), self.goal,
                                     dynamic=soft)
            if path is not None:
                reason += "+softened"

        if path is None:
            self.log.reroute(reason + ":FAILED", attempt=self.reroute_count)
            return False

        self.path = path
        self.follower.reset()
        self.publish_path()
        self.log.reroute(reason, old_len=old_len,
                         new_len=Planner.path_length(path),
                         attempt=self.reroute_count)
        return True

    def publish_path(self):
        msg = Path()
        msg.header.frame_id = self.path_frame
        msg.header.stamp = self.get_clock().now().to_msg()
        for x, y in self.path:
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x = float(x)
            ps.pose.position.y = float(y)
            ps.pose.orientation.w = 1.0
            msg.poses.append(ps)
        self.pub_path.publish(msg)

    def publish_intent(self):
        """Broadcast what we are trying to do, for the collision layer.

        The collision node owns the /fleet/state broadcast because it owns
        the kinematics; it gets our goal and task from here so that peers
        can see intent, not just motion.
        """
        self.pub_intent.publish(encode({
            "goal": [round(self.goal[0], 2), round(self.goal[1], 2)]
                    if self.goal else None,
            "task": self.task_id,
            "phase": self.phase,
            "nav_state": self.state,
        }))

    # ----------------------------------------------------------- main loop

    def tick(self):
        if self.pose is None:
            return
        self.peers.prune(self.now())

        if self.state == IDLE:
            self.publish(0.0, 0.0)
            return
        if self.state == PLANNING:
            self.do_planning()
        elif self.state == FOLLOWING:
            self.do_following()
        elif self.state == BACKOFF:
            self.do_backoff()
        elif self.state == WAITING:
            self.do_waiting()

    # ------------------------------------------------------------- states

    def do_planning(self):
        self.publish(0.0, 0.0)
        if self.replan("plan"):
            self.enter(FOLLOWING)
        else:
            self.enter(WAITING)

    def do_following(self):
        now = self.now()
        if len(self.path) < 2:
            self.enter(PLANNING)
            return

        # An escalation from the collision layer, or a peer that has parked
        # itself on our route, both mean the same thing: this path is no
        # longer the right one.
        want_replan = self.pending_reroute
        if not want_replan and self.planner is not None:
            if now - self.last_plan_t > self.replan_period:
                want_replan = not self.planner.path_is_clear(
                    self.path, self.dynamic_layer(), self.follower.index,
                    look_ahead_m=6.0)
                self.last_plan_t = now

        if want_replan:
            self.pending_reroute = False
            self.reroute_count += 1
            if self.reroute_count > self.max_reroutes:
                self.begin_backoff("reroute limit reached")
                return
            if not self.replan("reroute#%d" % self.reroute_count):
                self.begin_backoff("no alternative route")
                return

        result = self.follower.step(self.pose, self.speed, self.path,
                                    self.goal, self.cmd_v, self.dt)
        if result.arrived:
            self.publish(0.0, 0.0)
            err = math.hypot(self.goal[0] - self.pose[0],
                             self.goal[1] - self.pose[1])
            self.log.task("arrived", self.task_id or "manual",
                          "phase=%s err=%.2fm" % (self.phase, err))
            self.report("ARRIVED", "err=%.2f" % err)
            self.abandon("arrived")
            return

        self.publish(result.v, result.w)

    def begin_backoff(self, reason):
        """Retreat far enough to let the other robot through.

        Reversing is the one manoeuvre guaranteed to be permitted while a
        head-on conflict is active: it increases clearance, so the predictive
        layer downstream does not brake it. Note the direction of that
        argument -- the retreat is safe *because* the collision layer is
        predictive rather than distance-based.

        Retreating repeatedly is not a strategy, though. After
        ``max_backoffs`` attempts the honest conclusion is that this robot
        cannot serve this task right now, and the task goes back to the
        fleet for someone better placed to pick up.
        """
        self.backoffs += 1
        if self.backoffs > self.max_backoffs:
            self.log.task("blocked", self.task_id or "manual",
                          "gave up after %d retreats" % (self.backoffs - 1))
            self.report("BLOCKED", "backoffs=%d" % (self.backoffs - 1))
            self.abandon("blocked")
            return
        self.backoff_start = (self.pose[0], self.pose[1])
        self.backoff_reason = reason
        self.enter(BACKOFF)
        self.log.reroute("backoff:" + reason, attempt=self.reroute_count)

    def do_backoff(self):
        travelled = math.hypot(self.pose[0] - self.backoff_start[0],
                               self.pose[1] - self.backoff_start[1])
        elapsed = self.now() - self.state_since
        if travelled >= self.backoff_distance or elapsed > self.backoff_timeout:
            self.publish(0.0, 0.0)
            self.reroute_count = 0
            self.enter(PLANNING)
            return
        self.publish(-self.backoff_speed, 0.0)

    def do_waiting(self):
        """No route right now. Wait, retry, and eventually give the task back."""
        self.publish(0.0, 0.0)
        waited = self.now() - self.state_since

        if waited > self.wait_timeout:
            self.log.task("unreachable", self.task_id or "manual",
                          "waited=%.1fs" % waited)
            self.report("UNREACHABLE", "waited=%.1f" % waited)
            self.abandon("unreachable")
            return

        if self.now() - self.last_plan_t > 1.5:
            # Peers move; a route that did not exist a second ago often does
            # now, which is why this retries instead of failing immediately.
            if self.replan("retry"):
                self.reroute_count = 0
                self.enter(FOLLOWING)

    # -------------------------------------------------------------- output

    def publish(self, v, w):
        self.cmd_v = v
        msg = Twist()
        msg.linear.x = float(v)
        msg.angular.z = float(w)
        self.pub_cmd.publish(msg)


def main():
    rclpy.init()
    node = FleetNavigator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.pub_cmd.publish(Twist())
        node.log.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
