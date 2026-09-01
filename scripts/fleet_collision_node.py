#!/usr/bin/env python3
"""Decentralized collision avoidance and conflict resolution for one AMR.

One instance runs per robot. There is no central arbiter: each robot
broadcasts its own kinematic state on ``/fleet/state``, listens to every
peer, and resolves conflicts locally with rules that are *symmetric* -- the
peer running this same file reaches the inverse conclusion from the same
numbers, so no negotiation round-trip exists to go wrong.

Where this sits::

    fleet_navigator --> cmd_vel_nav --> [ THIS NODE ] --> cmd_vel --> base

The node scales the incoming command rather than replacing it, so whatever
is upstream -- this package navigator, or Nav2 controller_server -- keeps
full authority over *steering*, and this layer only decides *whether* and
*how fast*. It is also the single place in the stack that can zero the base
command, which is what makes the safety argument reviewable.

Why prediction rather than distance
-----------------------------------
The rule this replaces triggered on raw separation. ``dist=1.20m`` carries
no information about intent: it is the same number whether two robots are
closing head-on at 0.5 m/s or driving apart. In the 09:16 encounter that
produced 80 braking samples for a pass whose closest approach was never
going to be tighter than 0.88 m, including 40 samples taken while the two
robots were actively separating. The predictive test asks a different
question -- given both velocities, do these two chassis ever come within a
safety margin of each other inside the horizon -- and the answer is False
for every one of those samples.

Four layers, cheapest first:

    predictive (CPA + swept footprint)   conflicts visible 6-8 s out
    arbitration (aged right of way)      symmetric, starvation-free
    speed governor                       continuous, not binary stop
    unconditional floors                 hull clearance, and the laser

Each layer can only ever reduce speed, so the safety argument is a floor,
not a negotiation.
"""

import json
import math

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Point, Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, ColorRGBA, String
from visualization_msgs.msg import Marker, MarkerArray

from fleet_coordination.avoidance import ConflictResolver, PeerView
from fleet_coordination.bus import (FLEET_STATE_TOPIC, FleetState, PeerTable,
                                    encode, state_qos)
from fleet_coordination.flog import FleetLogger
from fleet_coordination.geometry import capsule_clearance, yaw_from_quat

MARKER_TOPIC = "/fleet/markers"


class FleetCollisionNode(Node):

    def __init__(self):
        super().__init__("fleet_collision_node")

        self.declare_parameters("", [
            # ---- identity ----
            ("robot_id", "amr1"),
            ("base_priority", 1.0),

            # ---- chassis ----
            # Measured off bcr_bot.xacro: chassis_length 0.90, width 0.64.
            # Modelled as a capsule, which is the exact rounded hull of that
            # rectangle -- see fleet_coordination.geometry.Footprint.
            ("footprint_length", 0.90),
            ("footprint_width", 0.64),

            # ---- clearances, in metres of air between hulls ----
            ("safety_margin", 0.25),
            ("hysteresis_m", 0.20),
            ("emergency_clearance", 0.15),

            # ---- prediction ----
            ("horizon_s", 8.0),
            ("sweep_step_s", 0.25),
            ("brake_ttc_s", 2.5),
            ("stop_ttc_s", 1.0),
            ("min_speed_scale", 0.15),
            ("row_speed_scale", 0.85),

            # ---- arbitration ----
            ("aging_rate", 0.25),
            ("arbitration_deadband", 0.05),
            ("max_wait_credit", 30.0),
            ("latch_release_s", 1.0),
            ("headon_angle_deg", 135.0),
            ("overtake_angle_deg", 45.0),
            ("still_speed", 0.05),

            # ---- deadlock escalation ----
            ("max_hold_s", 6.0),
            ("headon_hold_s", 1.5),
            ("blocked_hold_s", 2.5),
            ("reroute_repeat_s", 8.0),

            # ---- laser floor ----
            ("laser_enabled", True),
            ("laser_stop_m", 0.70),
            ("laser_cone_deg", 50.0),
            ("laser_peer_tolerance_m", 0.30),

            # ---- plumbing ----
            ("peer_timeout_s", 1.5),
            ("control_hz", 20.0),
            ("broadcast_hz", 10.0),
            ("cmd_timeout_s", 0.75),
            ("rotation_margin_m", 0.10),
            ("publish_markers", True),
            ("marker_frame", "map"),
            ("log_file", ""),
            ("log_period_s", 0.5),
        ])

        g = self.get_parameter
        self.rid = g("robot_id").value
        self.base_priority = float(g("base_priority").value)

        # All peer reasoning lives in the resolver, which knows nothing
        # about ROS -- the offline simulator drives this same class, so what
        # is validated there is what runs here.
        self.resolver = ConflictResolver(
            self.rid, self.base_priority,
            footprint_length=float(g("footprint_length").value),
            footprint_width=float(g("footprint_width").value),
            safety_margin=float(g("safety_margin").value),
            hysteresis=float(g("hysteresis_m").value),
            emergency_clearance=float(g("emergency_clearance").value),
            horizon=float(g("horizon_s").value),
            sweep_step=float(g("sweep_step_s").value),
            brake_ttc=float(g("brake_ttc_s").value),
            stop_ttc=float(g("stop_ttc_s").value),
            min_scale=float(g("min_speed_scale").value),
            row_scale=float(g("row_speed_scale").value),
            headon_deg=float(g("headon_angle_deg").value),
            overtake_deg=float(g("overtake_angle_deg").value),
            still_speed=float(g("still_speed").value),
            aging_rate=float(g("aging_rate").value),
            deadband=float(g("arbitration_deadband").value),
            max_wait_credit=float(g("max_wait_credit").value),
            latch_release_s=float(g("latch_release_s").value))
        self.fp = self.resolver.fp

        self.max_hold_s = float(g("max_hold_s").value)
        self.headon_hold_s = float(g("headon_hold_s").value)
        self.blocked_hold_s = float(g("blocked_hold_s").value)
        self.reroute_repeat_s = float(g("reroute_repeat_s").value)

        self.laser_enabled = bool(g("laser_enabled").value)
        self.laser_stop_m = float(g("laser_stop_m").value)
        self.laser_cone = math.radians(float(g("laser_cone_deg").value)) * 0.5
        self.laser_tol = float(g("laser_peer_tolerance_m").value)

        self.cmd_timeout = float(g("cmd_timeout_s").value)
        self.rotation_margin = float(g("rotation_margin_m").value)
        self.publish_markers = bool(g("publish_markers").value)
        self.marker_frame = g("marker_frame").value
        self.log_period = float(g("log_period_s").value)

        self.dt = 1.0 / max(1.0, float(g("control_hz").value))
        broadcast_dt = 1.0 / max(1.0, float(g("broadcast_hz").value))

        self.peers = PeerTable(timeout_s=float(g("peer_timeout_s").value))

        # ---- state ----
        self.pose = None
        self.vel = (0.0, 0.0, 0.0)       # world frame (vx, vy, omega)
        self.speed = 0.0
        self.cmd = Twist()
        self.cmd_stamp = -1e9
        self.laser_min = float("inf")
        self.intent = {}                 # goal / task / phase from the navigator

        self.wait_s = 0.0                # live yield credit
        self.wait_broadcast = 0.0        # the value peers arbitrate against
        self.hold_s = 0.0
        self.engaged = set()             # peers we are currently in conflict with
        self.status = "CLEAR"
        self.scale = 1.0
        self.blocker = None
        self.last_reroute_t = -1e9
        self.seq = 0
        self._reason_timers = {}

        self.log = FleetLogger(self.rid, path=(g("log_file").value or None),
                               sim_clock=self.now, ros_logger=self.get_logger())

        # ---- ROS interfaces ----
        qos = state_qos()
        self.create_subscription(Odometry, "odom", self.on_odom, qos)
        self.create_subscription(Twist, "cmd_vel_nav", self.on_cmd, 10)
        self.create_subscription(String, "intent", self.on_intent, 10)
        self.create_subscription(String, FLEET_STATE_TOPIC, self.on_peer, qos)
        if self.laser_enabled:
            self.create_subscription(LaserScan, "scan", self.on_scan, qos)

        self.pub_cmd = self.create_publisher(Twist, "cmd_vel", 10)
        self.pub_state = self.create_publisher(String, FLEET_STATE_TOPIC, qos)
        self.pub_status = self.create_publisher(String, "avoidance_status", 10)
        self.pub_reroute = self.create_publisher(Bool, "reroute_request", 10)
        self.pub_conflict = self.create_publisher(String, "conflict_report", 10)
        self.pub_markers = (self.create_publisher(MarkerArray, MARKER_TOPIC, 5)
                            if self.publish_markers else None)

        self.create_timer(self.dt, self.tick)
        self.create_timer(broadcast_dt, self.broadcast)

        self.get_logger().info(
            "[%s] avoidance up | %s | margin=%.2fm emergency=%.2fm "
            "horizon=%.0fs prio=%.1f"
            % (self.rid, self.fp, self.resolver.margin, self.resolver.emergency,
               self.resolver.horizon, self.base_priority))

    # ------------------------------------------------------------- clock

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    # --------------------------------------------------------------- I/O

    def on_odom(self, msg):
        p = msg.pose.pose.position
        yaw = yaw_from_quat(msg.pose.pose.orientation)
        self.pose = (p.x, p.y, yaw)
        # odom twist is body frame; rotate the linear part into the world.
        vx_b = msg.twist.twist.linear.x
        vy_b = msg.twist.twist.linear.y
        c, s = math.cos(yaw), math.sin(yaw)
        self.vel = (vx_b * c - vy_b * s,
                    vx_b * s + vy_b * c,
                    msg.twist.twist.angular.z)
        self.speed = math.hypot(self.vel[0], self.vel[1])

    def on_cmd(self, msg):
        self.cmd = msg
        self.cmd_stamp = self.now()

    def on_intent(self, msg):
        try:
            self.intent = json.loads(msg.data)
        except (ValueError, TypeError):
            self.intent = {}

    def on_peer(self, msg):
        try:
            payload = json.loads(msg.data)
        except (ValueError, TypeError):
            return
        if payload.get("id") == self.rid:
            return
        self.peers.update(payload, self.now())

    def on_scan(self, msg):
        """Narrow forward cone, minimum valid range. A floor, nothing more."""
        best = float("inf")
        n = len(msg.ranges)
        if not n:
            self.laser_min = best
            return
        for i, r in enumerate(msg.ranges):
            if not (msg.range_min <= r <= msg.range_max):
                continue                       # inf, nan, or inside the blind spot
            ang = msg.angle_min + i * msg.angle_increment
            if abs(math.atan2(math.sin(ang), math.cos(ang))) > self.laser_cone:
                continue
            if r < best:
                best = r
        self.laser_min = best

    # --------------------------------------------------------- broadcast

    def broadcast(self):
        if self.pose is None:
            return
        self.wait_broadcast = self.wait_s
        self.seq += 1
        goal = self.intent.get("goal")
        self.pub_state.publish(encode(FleetState.payload(
            self.rid, self.now(), self.pose, self.vel, self.speed,
            self.base_priority, self.wait_broadcast, self.status,
            task_id=self.intent.get("task"),
            phase=self.intent.get("phase", "IDLE"),
            goal=goal, seq=self.seq)))

    # ------------------------------------------------------- classification

    def desired_twist(self, cmd_fresh):
        """The motion we would make if nothing were in the way.

        Predicting from *measured* velocity is self-defeating for a robot
        that is already yielding: once it stops, the prediction says there is
        no longer a conflict, so it releases, accelerates, re-detects, and
        brakes again -- a limit cycle that looks exactly like the 4.5 s
        lockstep in the 09:18 log. Predicting from the *commanded* velocity
        asks the stable question instead: would there be a conflict if I
        proceeded?
        """
        if not cmd_fresh:
            return self.vel
        v = self.cmd.linear.x
        w = self.cmd.angular.z
        if abs(v) < 1e-3 and abs(w) < 1e-3:
            return self.vel
        yaw = self.pose[2]
        return (v * math.cos(yaw), v * math.sin(yaw), w)

    def peer_views(self):
        """Adapt the /fleet/state table into the resolver own view type."""
        return [PeerView(p.rid, p.pose, p.twist, p.speed, p.priority,
                         p.wait, p.state) for p in self.peers.values()]

    # ----------------------------------------------------------- main loop

    def tick(self):
        if self.pose is None:
            return

        now = self.now()
        for dead in self.peers.prune(now):
            self.engaged.discard(dead)

        cmd_fresh = (now - self.cmd_stamp) < self.cmd_timeout
        my_twist = self.desired_twist(cmd_fresh)
        peers = self.peer_views()

        decision = self.resolver.resolve(
            now, self.pose, my_twist, self.speed, self.wait_broadcast,
            peers, engaged_prev=self.engaged)

        scale, status = decision.scale, decision.status
        blocker, reason = decision.blocker, decision.reason
        self.engaged = decision.engaged

        for rep in decision.reports:
            self.maybe_log_conflict(now, rep)

        # ---- laser floor: anything the map and the fleet bus did not know ----
        if self.laser_enabled and self.laser_stop_m > 0.0 and self.laser_blocked():
            scale, status = 0.0, "LASER_STOP"
            reason = reason or "laser"

        # ---- turning on the spot is a separate permission from driving ----
        translating = cmd_fresh and (abs(self.cmd.linear.x) > 1e-2
                                     or abs(self.cmd.linear.y) > 1e-2)
        rotating = cmd_fresh and abs(self.cmd.angular.z) > 1e-3
        if (not translating and rotating
                and status not in ("EMERGENCY_STOP", "LASER_STOP")
                and not self.resolver.rotation_is_safe(
                    self.pose, peers, self.rotation_margin)):
            # Otherwise this stalls silently: scale stays 1.0, the deadlock
            # clock never starts, and nobody ever asks for a reroute.
            scale, status = 0.0, "ROTATE_BLOCKED"
            reason = reason or "blocked"
            if blocker is None:
                blocker = self.nearest_peer_id()

        self.bookkeeping(now, scale, status, blocker, reason)
        self.emit(scale, status, translating, cmd_fresh)
        if self.pub_markers is not None:
            self.publish_marker_array(decision.reports, status, scale)

    def nearest_peer_id(self):
        best, best_d = None, float("inf")
        for rid, peer in self.peers.items():
            d = math.hypot(peer.x - self.pose[0], peer.y - self.pose[1])
            if d < best_d:
                best, best_d = rid, d
        return best

    def laser_blocked(self):
        """True when the laser sees something closer than any known peer.

        Peers are already handled predictively and far more precisely than a
        range bin can manage; letting the laser brake for them as well would
        double-count every legitimate pass. So the floor only fires for
        returns that no peer can account for.
        """
        if self.laser_min > self.laser_stop_m:
            return False
        nearest_peer = min(
            (capsule_clearance(self.fp, self.pose, self.fp, p.pose)
             for p in self.peers.values()), default=float("inf"))
        return self.laser_min < nearest_peer - self.laser_tol

    # ------------------------------------------------------- bookkeeping

    def bookkeeping(self, now, scale, status, blocker, reason):
        """Wait credit, deadlock timers, and the decision to ask for a reroute."""
        # Anything that held us up on account of a *peer* earns wait credit,
        # because that credit is what buys back right of way. A laser stop
        # does not: no peer caused it, and no peer should pay for it.
        yielding = (status.startswith("YIELD") or status.startswith("STOP")
                    or status.startswith("SLOW")
                    or status in ("EMERGENCY_STOP", "ROTATE_BLOCKED"))

        if yielding:
            self.wait_s += self.dt
        else:
            self.wait_s = max(0.0, self.wait_s - 2.0 * self.dt)

        if scale <= 0.0:
            self.hold_s += self.dt
        else:
            self.hold_s = 0.0

        # Per-reason escalation clocks. Head-on and "peer is parked in my
        # way" are known-unresolvable by waiting, so they escalate in a
        # couple of seconds; a plain crossing conflict gets the full
        # deadlock timeout because waiting usually does resolve it.
        limit = {"head-on": self.headon_hold_s,
                 "blocked": self.blocked_hold_s}.get(reason, self.max_hold_s)

        if self.hold_s >= limit and blocker is not None:
            if now - self.last_reroute_t >= self.reroute_repeat_s:
                self.last_reroute_t = now
                self.request_reroute(now, blocker, reason or "deadlock")

        if status != self.status:
            self.get_logger().info(
                "[%s] %s -> %s%s" % (self.rid, self.status, status,
                                     "" if blocker is None
                                     else " (peer %s, scale %.2f)" % (blocker, scale)))
        self.status = status
        self.scale = scale
        self.blocker = blocker

    def request_reroute(self, now, blocker, reason):
        peer = self.peers.get(blocker)
        self.pub_reroute.publish(Bool(data=True))
        self.pub_conflict.publish(encode({
            "id": self.rid,
            "t": round(now, 3),
            "reason": reason,
            "peer": blocker,
            "peer_pos": [round(peer.x, 2), round(peer.y, 2)] if peer else None,
            "peer_state": peer.state if peer else None,
            "held_s": round(self.hold_s, 2),
        }))
        self.log.warn("held %.1fs behind %s (%s) -- requesting reroute"
                      % (self.hold_s, blocker, reason))

    def maybe_log_conflict(self, now, rep):
        # Throttled per peer, not globally: with three peers a global
        # throttle would silently drop two thirds of the encounters.
        if now - self._reason_timers.get(rep.peer.rid, -1e9) < self.log_period:
            return
        self._reason_timers[rep.peer.rid] = now

        peer = rep.peer
        dist = math.hypot(peer.x - self.pose[0], peer.y - self.pose[1])
        self.log.proximity(
            peer.rid, dist, rep.gap_now,
            {"pos": self.pose, "yaw": self.pose[2], "vel": self.vel,
             "prio": self.base_priority, "goal": self.intent.get("goal"),
             "task": self.intent.get("task")},
            {"pos": (peer.x, peer.y), "yaw": peer.pose[2],
             "vel": (peer.twist[0], peer.twist[1]),
             "prio": peer.priority, "goal": None, "task": None})
        self.log.conflict(
            peer.rid,
            rep.t_cpa if rep.t_cpa != math.inf else 999.0, rep.d_cpa,
            rep.t_hit, rep.gap_min, rep.kind,
            self.rid if rep.i_have_row else peer.rid, rep.status, rep.scale)

    # -------------------------------------------------------------- output

    def emit(self, scale, status, translating, cmd_fresh):
        out = Twist()
        cmd = self.cmd

        if not cmd_fresh:
            # Upstream went quiet. A stale command is worse than no command.
            self.pub_cmd.publish(out)
            self.pub_status.publish(
                String(data="NO_CMD|0.00|%s" % (self.blocker or "-")))
            return

        if status in ("EMERGENCY_STOP", "LASER_STOP", "ROTATE_BLOCKED"):
            pass                                   # everything stays zero
        elif not translating:
            # Pure rotation. It does not close distance, and its safety was
            # already decided in tick() -- this is how a stopped robot turns
            # onto the detour its planner just produced.
            out.angular.z = cmd.angular.z
        else:
            # Scale linear and angular together: that traverses the same
            # geometric path more slowly. Scaling linear alone would tighten
            # every turn and walk the robot off its planned route.
            out.linear.x = cmd.linear.x * scale
            out.linear.y = cmd.linear.y * scale
            out.angular.z = cmd.angular.z * scale

        self.pub_cmd.publish(out)
        self.pub_status.publish(
            String(data="%s|%.2f|%s" % (status, scale, self.blocker or "-")))

    # ------------------------------------------------------------- markers

    def publish_marker_array(self, reports, status, scale):
        arr = MarkerArray()
        now = self.get_clock().now().to_msg()

        label = Marker()
        label.header.frame_id = self.marker_frame
        label.header.stamp = now
        label.ns = self.rid
        label.id = 0
        label.type = Marker.TEXT_VIEW_FACING
        label.action = Marker.ADD
        label.pose.position.x = self.pose[0]
        label.pose.position.y = self.pose[1]
        label.pose.position.z = 1.1
        label.pose.orientation.w = 1.0
        label.scale.z = 0.28
        label.color = self._colour(status)
        label.text = "%s %s %.2f" % (self.rid, status, scale)
        arr.markers.append(label)

        for i, rep in enumerate(reports, start=1):
            link = Marker()
            link.header.frame_id = self.marker_frame
            link.header.stamp = now
            link.ns = self.rid
            link.id = i
            link.type = Marker.LINE_LIST
            link.action = Marker.ADD
            link.scale.x = 0.05
            link.color = self._colour(rep.status)
            link.pose.orientation.w = 1.0
            link.points = [Point(x=self.pose[0], y=self.pose[1], z=0.35),
                           Point(x=rep.peer.x, y=rep.peer.y, z=0.35)]
            arr.markers.append(link)

        # Retire markers left over from a tick with more conflicts than this one.
        for i in range(len(reports) + 1, len(reports) + 5):
            gone = Marker()
            gone.header.frame_id = self.marker_frame
            gone.ns = self.rid
            gone.id = i
            gone.action = Marker.DELETE
            arr.markers.append(gone)

        self.pub_markers.publish(arr)

    @staticmethod
    def _colour(status):
        if status in ("EMERGENCY_STOP", "LASER_STOP"):
            return ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)
        if status.endswith("STOP") or status == "YIELD_HEADON":
            return ColorRGBA(r=1.0, g=0.35, b=0.0, a=1.0)
        if status.startswith("YIELD") or status.startswith("SLOW"):
            return ColorRGBA(r=1.0, g=0.85, b=0.0, a=1.0)
        if status == "PROCEED_ROW":
            return ColorRGBA(r=0.2, g=0.6, b=1.0, a=1.0)
        return ColorRGBA(r=0.2, g=0.9, b=0.3, a=1.0)


def main():
    rclpy.init()
    node = FleetCollisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.pub_cmd.publish(Twist())          # stop the base on the way out
        node.log.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
