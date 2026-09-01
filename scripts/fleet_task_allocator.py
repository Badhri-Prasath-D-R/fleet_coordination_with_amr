#!/usr/bin/env python3
"""Decentralized task allocation for one AMR: bid, win, execute, hand back.

Every robot runs this identical node. Work is announced on ``/fleet/market``,
every robot with spare capacity prices it and broadcasts a bid, and when the
bid window closes each robot independently computes the same winner from the
same bids. There is no allocator process -- nothing to elect, nothing to fail
over, nothing to bottleneck on. See :mod:`fleet_coordination.tasks` for why
that converges even when messages go missing.

The interesting property is the bid itself. Robots bid *estimated seconds
until the job is finished*, computed over a real A* route rather than a
straight line, and including whatever they are already carrying. That single
choice does most of the load balancing for free:

* a robot standing on the pickup but with two jobs queued honestly bids worse
  than an idle robot six metres away;
* a robot whose route to the pickup is blocked by shelving bids the length of
  the detour, not the length of the crow-flight;
* a robot that cannot route there at all does not bid, rather than winning a
  job it will fail.

This node also closes the loop with the rest of the stack. When the navigator
reports that it has exhausted its reroutes and retreats, the task is
*released* back to the market instead of being retried forever -- so a
blocked aisle costs the fleet one reassignment, not one stalled robot.
"""

import json
import math
import os

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from std_msgs.msg import String

from fleet_coordination.bus import (FLEET_EVENT_TOPIC, FLEET_MARKET_TOPIC,
                                    FLEET_STATE_TOPIC, FLEET_TASK_TOPIC,
                                    PeerTable, encode, event_qos, market_qos,
                                    state_qos, task_qos)
from fleet_coordination.flog import FleetLogger
from fleet_coordination.geometry import yaw_from_quat
from fleet_coordination.gridmap import load_map
from fleet_coordination.planner import Planner
from fleet_coordination.tasks import (DONE, Market, Task, bid_cost,
                                      msg_announce, msg_award, msg_bid,
                                      msg_done, msg_progress, msg_release)


class FleetTaskAllocator(Node):

    def __init__(self):
        super().__init__("fleet_task_allocator")

        self.declare_parameters("", [
            ("robot_id", "amr1"),
            ("map_yaml", ""),
            ("planning_downsample", 6),
            ("planning_clearance", 0.45),

            # ---- auction ----
            ("bid_window_s", 1.0),
            ("owner_timeout_s", 4.0),
            ("assign_timeout_s", 8.0),
            ("reopen_period_s", 5.0),
            ("max_attempts", 4),
            ("max_queue", 2),

            # ---- bid shape ----
            ("nominal_speed", 0.35),
            ("queue_penalty_s", 8.0),
            ("congestion_penalty_s", 6.0),
            ("congestion_radius_m", 3.0),

            # ---- execution ----
            ("dwell_s", 2.0),
            ("tick_hz", 5.0),
            ("peer_timeout_s", 2.0),
            ("log_file", ""),
        ])

        g = self.get_parameter
        self.rid = g("robot_id").value
        self.max_queue = int(g("max_queue").value)
        self.nominal_speed = float(g("nominal_speed").value)
        self.queue_penalty = float(g("queue_penalty_s").value)
        self.congestion_penalty = float(g("congestion_penalty_s").value)
        self.congestion_radius = float(g("congestion_radius_m").value)
        self.dwell_s = float(g("dwell_s").value)

        self.market = Market(
            self.rid,
            bid_window_s=float(g("bid_window_s").value),
            owner_timeout_s=float(g("owner_timeout_s").value),
            reopen_period_s=float(g("reopen_period_s").value),
            max_attempts=int(g("max_attempts").value),
            assign_timeout_s=float(g("assign_timeout_s").value))

        self.log = FleetLogger(self.rid, path=(g("log_file").value or None),
                               sim_clock=self.now, ros_logger=self.get_logger())

        # ---- routing, used only for pricing bids ----
        self.planner = None
        map_yaml = g("map_yaml").value
        if map_yaml and os.path.exists(os.path.expanduser(map_yaml)):
            grid = load_map(os.path.expanduser(map_yaml),
                            downsample=int(g("planning_downsample").value))
            self.planner = Planner(
                grid, min_clearance=float(g("planning_clearance").value))
            self.get_logger().info("[%s] bid pricing over %s" % (self.rid, grid))
        else:
            self.get_logger().warn(
                "[%s] no map -- bids priced on straight-line distance"
                % self.rid)

        # ---- state ----
        self.pose = None
        self.peers = PeerTable(timeout_s=float(g("peer_timeout_s").value))
        self.queue = []                  # task ids we have won, in order
        self.current = None              # task id being executed
        self.phase = "IDLE"              # TO_PICKUP / DWELL_PICKUP / TO_DROPOFF
        self.dwell_until = 0.0
        self.completed = 0

        # ---- ROS interfaces ----
        self.create_subscription(Odometry, "odom", self.on_odom, state_qos())
        self.create_subscription(String, FLEET_STATE_TOPIC, self.on_peer,
                                 state_qos())
        self.create_subscription(String, FLEET_TASK_TOPIC, self.on_task_request,
                                 task_qos())
        self.create_subscription(String, FLEET_MARKET_TOPIC, self.on_market,
                                 market_qos())
        self.create_subscription(String, "nav_result", self.on_nav_result, 10)

        self.pub_market = self.create_publisher(String, FLEET_MARKET_TOPIC,
                                                market_qos())
        self.pub_events = self.create_publisher(String, FLEET_EVENT_TOPIC,
                                                event_qos())
        self.pub_goal = self.create_publisher(String, "nav_goal", 10)

        self.create_timer(1.0 / max(1.0, float(g("tick_hz").value)), self.tick)

        self.get_logger().info(
            "[%s] allocator up | max_queue=%d bid_window=%.1fs"
            % (self.rid, self.max_queue, self.market.bid_window))

    # ------------------------------------------------------------- clock

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    # ---------------------------------------------------------------- I/O

    def on_odom(self, msg):
        p = msg.pose.pose.position
        self.pose = (p.x, p.y, yaw_from_quat(msg.pose.pose.orientation))

    def on_peer(self, msg):
        try:
            payload = json.loads(msg.data)
        except (ValueError, TypeError):
            return
        if payload.get("id") == self.rid:
            return
        self.peers.update(payload, self.now())

    def on_task_request(self, msg):
        """New work entering the fleet. Everyone hears it; everyone prices it."""
        try:
            d = json.loads(msg.data)
        except (ValueError, TypeError):
            return
        try:
            task = Task.from_dict(d if "tid" in d else d.get("task", {}))
        except (KeyError, TypeError, ValueError):
            self.get_logger().warn("[%s] malformed task request" % self.rid)
            return
        if not task.created:
            task.created = self.now()
        if self.market.announce(task, self.now()):
            self.event("task_announced", tid=task.tid, label=task.label)
            self.place_bid(task)

    def on_market(self, msg):
        try:
            d = json.loads(msg.data)
        except (ValueError, TypeError):
            return
        if d.get("from") == self.rid:
            return                       # our own traffic is applied locally
        self.apply(d)

    def on_nav_result(self, msg):
        """The navigator finished, or gave up."""
        try:
            d = json.loads(msg.data)
        except (ValueError, TypeError):
            return
        tid = d.get("task")
        if tid is None or tid != self.current:
            return

        result = d.get("result")
        if result == "ARRIVED":
            self.on_arrival()
        elif result in ("UNREACHABLE", "BLOCKED"):
            self.release_current(result.lower())

    # -------------------------------------------------------- market glue

    def emit(self, payload):
        """Publish onto the market and apply the same message to ourselves.

        Applying locally rather than relying on hearing our own publication
        back means the protocol behaves identically whether or not the RMW
        loops messages back to their sender.
        """
        self.pub_market.publish(encode(payload))
        self.apply(payload)

    def apply(self, d):
        kind = d.get("type")
        now = self.now()
        if kind == "ANNOUNCE":
            task = Task.from_dict(d["task"])
            if self.market.announce(task, now):
                self.place_bid(self.market.tasks[task.tid])
        elif kind == "BID":
            self.market.record_bid(d["tid"], d["from"], d["cost"], now)
        elif kind == "AWARD":
            if self.market.apply_award(d["tid"], d["winner"], d["cost"], now):
                self.on_award(d["tid"], d["winner"])
        elif kind == "PROGRESS":
            self.market.apply_progress(d["tid"], d["from"], d["phase"], now)
        elif kind == "DONE":
            self.market.apply_done(d["tid"], d["from"], now)
            self.drop_local(d["tid"])
        elif kind == "RELEASE":
            self.market.apply_release(d["tid"], d["from"], now)
            self.drop_local(d["tid"])

    def on_award(self, tid, winner):
        """Our view of an owner changed. Adopt it, or let it go.

        The second half matters: a robot can lose a task it already started,
        because a peer that saw a bid we missed asserts a cheaper claim. The
        comparison is the same on both sides, so exactly one of us keeps it.
        """
        if winner == self.rid:
            if tid not in self.queue and tid != self.current:
                self.queue.append(tid)
                self.log.task("won", tid, "queue=%d" % len(self.queue))
                self.event("task_won", tid=tid)
        else:
            if tid == self.current:
                self.log.task("preempted", tid, "to=%s" % winner)
                self.event("task_preempted", tid=tid, to=winner)
                self.cancel_nav()
                self.current, self.phase = None, "IDLE"
            elif tid in self.queue:
                self.queue.remove(tid)

    def drop_local(self, tid):
        if tid in self.queue:
            self.queue.remove(tid)
        if tid == self.current:
            self.cancel_nav()
            self.current, self.phase = None, "IDLE"

    # ------------------------------------------------------------ bidding

    def route_len(self, a, b):
        if self.planner is not None:
            return self.planner.route_cost(a, b)
        return math.hypot(b[0] - a[0], b[1] - a[1]) * 1.3

    def busy_remaining(self):
        """Roughly how much driving is already on our plate, in metres."""
        if self.pose is None:
            return 0.0
        total = 0.0
        task = self.market.tasks.get(self.current) if self.current else None
        if task is not None:
            target = (task.dropoff if self.phase == "TO_DROPOFF"
                      else task.pickup)
            leg = self.route_len((self.pose[0], self.pose[1]), target)
            total += leg if leg is not None else 0.0
            if self.phase != "TO_DROPOFF":
                leg = self.route_len(task.pickup, task.dropoff)
                total += leg if leg is not None else 0.0
        for tid in self.queue:
            queued = self.market.tasks.get(tid)
            if queued is None:
                continue
            leg = self.route_len(queued.pickup, queued.dropoff)
            total += leg if leg is not None else 0.0
        return total

    def congestion_near(self, point):
        return sum(1 for p in self.peers.values()
                   if math.hypot(p.x - point[0], p.y - point[1])
                   < self.congestion_radius)

    def place_bid(self, task):
        if self.pose is None:
            return
        if self.rid in task.excluded:
            return                       # we already tried and could not
        load = len(self.queue) + (1 if self.current else 0)
        if load >= self.max_queue:
            return                       # honestly full; do not bid

        start = self.pose[0], self.pose[1]
        to_pickup = self.route_len(start, task.pickup)
        pickup_to_drop = self.route_len(task.pickup, task.dropoff)
        cost = bid_cost(to_pickup, pickup_to_drop, self.nominal_speed,
                        queue_len=load,
                        busy_remaining_m=self.busy_remaining(),
                        congestion=self.congestion_near(task.pickup),
                        queue_penalty_s=self.queue_penalty,
                        congestion_penalty_s=self.congestion_penalty)
        if cost is None:
            # Unroutable for us. Staying silent is the correct bid.
            self.log.task("no-bid", task.tid, "unroutable")
            return

        self.emit(msg_bid(task.tid, self.rid, cost, self.now()))
        self.log.task("bid", task.tid, "cost=%.1fs load=%d" % (cost, load))

    # -------------------------------------------------------------- ticks

    def tick(self):
        now = self.now()
        self.peers.prune(now)

        self.settle_auctions(now)
        self.housekeeping(now)
        self.drive_execution(now)
        self.market.forget_completed(now)

    def settle_auctions(self, now):
        """Close every bid window that has expired, identically on all robots."""
        for task in self.market.due_for_settlement(now):
            winner, cost = self.market.winner_of(task)
            if winner is None:
                self.market.settled.add(task.tid)
                continue
            if winner == self.rid:
                # We believe we won, so we say so. Peers that computed a
                # different winner reconcile against this claim.
                self.emit(msg_award(task.tid, winner, cost, self.rid, now))
                self.log.task("award", task.tid, "self cost=%.1fs" % cost)
            else:
                # Adopt provisionally; the winner own AWARD will confirm it,
                # and the assign timeout catches the case where it never does.
                if self.market.apply_award(task.tid, winner, cost, now):
                    self.on_award(task.tid, winner)

    def housekeeping(self, now):
        """Reopen work that has fallen on the floor -- one robot does this."""
        live = self.peers.ids()
        if not self.market.i_am_lowest_live(live):
            return
        for task in self.market.orphaned(live, now):
            was = task.owner
            if not self.market.reopen(task, now):
                self.log.task("failed", task.tid, "exhausted retries")
                self.event("task_failed", tid=task.tid)
                continue
            self.drop_local(task.tid)
            self.emit(msg_announce(task, self.rid, now))
            self.log.task("reopen", task.tid, "was=%s attempt=%d"
                          % (was, task.attempts))
            self.event("task_reopened", tid=task.tid, was=was)

    def drive_execution(self, now):
        if self.pose is None:
            return

        if self.current is None:
            if not self.queue:
                return
            self.current = self.queue.pop(0)
            self.phase = "TO_PICKUP"
            self.send_goal()
            return

        if self.phase == "DWELL_PICKUP" and now >= self.dwell_until:
            self.phase = "TO_DROPOFF"
            self.send_goal()
        elif self.phase == "DWELL_DROPOFF" and now >= self.dwell_until:
            self.finish_current()

    def send_goal(self):
        task = self.market.tasks.get(self.current)
        if task is None:
            self.current, self.phase = None, "IDLE"
            return
        target = task.pickup if self.phase == "TO_PICKUP" else task.dropoff
        self.pub_goal.publish(encode({
            "task": task.tid, "phase": self.phase,
            "x": target[0], "y": target[1]}))
        self.emit(msg_progress(task.tid, self.rid, self.phase, self.now()))
        self.log.task("start", task.tid, "phase=%s target=(%.2f,%.2f)"
                      % (self.phase, target[0], target[1]))
        self.event("task_progress", tid=task.tid, phase=self.phase)

    def cancel_nav(self):
        self.pub_goal.publish(encode({"cancel": True}))

    def on_arrival(self):
        now = self.now()
        if self.phase == "TO_PICKUP":
            self.phase = "DWELL_PICKUP"
            self.dwell_until = now + self.dwell_s
            self.event("task_picked", tid=self.current)
        elif self.phase == "TO_DROPOFF":
            self.phase = "DWELL_DROPOFF"
            self.dwell_until = now + self.dwell_s

    def finish_current(self):
        tid = self.current
        self.completed += 1
        self.emit(msg_done(tid, self.rid, self.now()))
        self.log.task("done", tid, "completed=%d" % self.completed)
        self.event("task_done", tid=tid, completed=self.completed)
        self.current, self.phase = None, "IDLE"

    def release_current(self, reason):
        """Hand a task we cannot serve back to the fleet.

        Note that we exclude ourselves from the next round. Without that,
        the same robot wins the same unreachable task again immediately and
        the fleet loops instead of adapting.
        """
        tid = self.current
        self.emit(msg_release(tid, self.rid, reason, self.now()))
        self.log.task("release", tid, "reason=%s" % reason)
        self.event("task_released", tid=tid, reason=reason)
        self.current, self.phase = None, "IDLE"

        task = self.market.tasks.get(tid)
        if task is not None and task.state not in (DONE,):
            self.emit(msg_announce(task, self.rid, self.now()))

    # ------------------------------------------------------------- events

    def event(self, kind, **fields):
        payload = {"t": round(self.now(), 3), "from": self.rid, "event": kind}
        payload.update(fields)
        self.pub_events.publish(encode(payload))


def main():
    rclpy.init()
    node = FleetTaskAllocator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.log.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
