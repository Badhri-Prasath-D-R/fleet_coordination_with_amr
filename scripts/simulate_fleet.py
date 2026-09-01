#!/usr/bin/env python3
"""Headless whole-fleet simulation. No ROS, no Gazebo, no build.

    python3 scripts/simulate_fleet.py
    python3 scripts/simulate_fleet.py --robots 4 --duration 400 --loss 0.1

What this actually validates matters, so it is worth being precise about it.
The simulator drives the *production* decision code:

    fleet_coordination.avoidance.ConflictResolver   the whole collision layer
    fleet_coordination.arbitration.RightOfWay       right of way and aging
    fleet_coordination.planner.Planner              A* and rerouting
    fleet_coordination.control.PathFollower         pure pursuit
    fleet_coordination.tasks.Market                 the auction

Those are the same objects the ROS nodes instantiate. What the simulator
supplies instead is the plumbing: a differential-drive integrator in place of
Gazebo, and a message bus in place of DDS. So a green run here is evidence
about the algorithms and their interaction, not about the rclpy wiring --
that still has to be seen in the sim.

Because the bus is synthetic, it can do things DDS will not do on request:
``--loss`` drops a configurable fraction of every broadcast, which is how the
arbitration deadband and the auction reconciliation rules get exercised
rather than merely asserted.

What it checks, and fails on:

    * hull overlap between any two robots at any tick (a real collision)
    * a robot stationary for longer than the starvation threshold
    * tasks that never complete
"""

import argparse
import math
import os
import random
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from fleet_coordination.avoidance import ConflictResolver, PeerView   # noqa: E402
from fleet_coordination.control import PathFollower                   # noqa: E402
from fleet_coordination.geometry import capsule_clearance, clamp, wrap_angle  # noqa: E402
from fleet_coordination.gridmap import load_map                       # noqa: E402
from fleet_coordination.planner import DynamicObstacle, Planner       # noqa: E402
from fleet_coordination.tasks import (DONE, Market, Task, bid_cost,   # noqa: E402
                                      msg_announce, msg_award, msg_bid,
                                      msg_done, msg_progress, msg_release)

START_POSES = {
    "amr1": (-4.50, -7.75, 0.0),
    "amr2": (-5.00, -3.50, 0.0),
    "amr3": (-2.00, -0.25, 0.0),
    "amr4": (-4.75, 1.25, 0.0),
}
BASE_PRIORITY = {"amr1": 1.0, "amr2": 2.0, "amr3": 3.0, "amr4": 4.0}


# ============================================================== the bus

class Bus:
    """A synthetic /fleet/* bus with a broadcast period and optional loss.

    State samples are held for one broadcast period before peers can see
    them. That one-period staleness is not incidental -- it is exactly the
    asymmetry the arbitration deadband exists to absorb, so simulating it
    faithfully is the point.
    """

    def __init__(self, rng, loss=0.0, broadcast_period=0.1):
        self.rng = rng
        self.loss = loss
        self.period = broadcast_period
        self.visible = {}          # rid -> last state peers can see
        self._pending = {}
        self._last_flush = -1e9
        self.market_log = []

    def broadcast_state(self, rid, sample):
        self._pending[rid] = sample

    def flush(self, now):
        if now - self._last_flush < self.period:
            return
        self._last_flush = now
        for rid, sample in self._pending.items():
            if self.loss and self.rng.random() < self.loss:
                continue           # dropped: peers keep the older sample
            self.visible[rid] = sample

    def peers_for(self, rid):
        return [s for r, s in self.visible.items() if r != rid]

    def publish_market(self, payload, robots):
        self.market_log.append(payload)
        for r in robots:
            if r.rid == payload["from"]:
                continue           # senders apply their own messages locally
            if self.loss and self.rng.random() < self.loss:
                continue
            r.apply_market(payload)


# ============================================================ one robot

class SimRobot:
    """One AMR: chassis integrator plus the three production decision layers.

    The state machines here mirror fleet_navigator.py and
    fleet_task_allocator.py. Everything they *decide* with -- conflicts,
    right of way, routes, bids -- is the shipped code, called directly.
    """

    def __init__(self, rid, grid, params, now=0.0):
        self.rid = rid
        x, y, yaw = START_POSES[rid]
        self.x, self.y, self.yaw = x, y, yaw
        self.v = 0.0
        self.w = 0.0

        self.planner = Planner(grid, min_clearance=params["planning_clearance"],
                               prefer_clearance=0.90)
        self.follower = PathFollower(
            max_linear=params["max_linear"], max_angular=params["max_angular"],
            max_accel=params["max_accel"])
        self.resolver = ConflictResolver(rid, BASE_PRIORITY[rid])
        self.market = Market(rid, bid_window_s=1.0, assign_timeout_s=8.0)

        self.params = params
        self.cmd_v = 0.0
        self.cmd_w = 0.0
        self.status = "CLEAR"
        self.scale = 1.0
        self.engaged = set()
        self.wait_s = 0.0
        self.wait_broadcast = 0.0
        self.hold_s = 0.0
        self.last_reroute = -1e9

        # navigator
        self.nav_state = "IDLE"
        self.goal = None
        self.path = []
        self.hard_blockers = {}
        self.pending_reroute = False
        self.reroutes = 0
        self.backoffs = 0
        self.backoff_from = None
        self.state_since = now
        self.last_plan = -1e9
        self.peer_still_since = {}

        # allocator
        self.queue = []
        self.current = None
        self.phase = "IDLE"
        self.dwell_until = 0.0
        self.completed = 0
        self.releases = 0
        self.last_alloc_tick = -1e9

        # metrics
        self.time_in_state = defaultdict(float)
        self.stopped_for = 0.0
        self.longest_stop = 0.0
        self.distance = 0.0

    # ------------------------------------------------------------ physics

    @property
    def pose(self):
        return (self.x, self.y, self.yaw)

    @property
    def twist(self):
        return (self.v * math.cos(self.yaw), self.v * math.sin(self.yaw), self.w)

    def integrate(self, dt):
        """Differential drive with acceleration limits. Gazebo stand-in."""
        max_a, max_alpha = 1.0, 3.0
        self.v = clamp(self.cmd_v, self.v - max_a * dt, self.v + max_a * dt)
        self.w = clamp(self.cmd_w, self.w - max_alpha * dt,
                       self.w + max_alpha * dt)
        self.yaw = wrap_angle(self.yaw + self.w * dt)
        step = self.v * dt
        self.x += step * math.cos(self.yaw)
        self.y += step * math.sin(self.yaw)
        self.distance += abs(step)

    def sample(self, now):
        return PeerView(self.rid, self.pose, self.twist, abs(self.v),
                        BASE_PRIORITY[self.rid], self.wait_broadcast,
                        self.status)

    # ----------------------------------------------------- market plumbing

    def emit(self, payload, bus, robots):
        self.apply_market(payload)
        bus.publish_market(payload, robots)

    def apply_market(self, d, now=None):
        now = self.now if now is None else now
        kind = d.get("type")
        if kind == "ANNOUNCE":
            task = Task.from_dict(d["task"])
            if self.market.announce(task, now):
                self._pending_bid.append(task.tid)
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
        if winner == self.rid:
            if tid not in self.queue and tid != self.current:
                self.queue.append(tid)
        else:
            if tid == self.current:
                self.current, self.phase = None, "IDLE"
                self.goal, self.path, self.nav_state = None, [], "IDLE"
            elif tid in self.queue:
                self.queue.remove(tid)

    def drop_local(self, tid):
        if tid in self.queue:
            self.queue.remove(tid)
        if tid == self.current:
            self.current, self.phase = None, "IDLE"
            self.goal, self.path, self.nav_state = None, [], "IDLE"

    # ------------------------------------------------------ allocator tick

    def allocator_tick(self, now, bus, robots, peers):
        if now - self.last_alloc_tick < 0.2:
            return
        self.last_alloc_tick = now

        for tid in self._pending_bid:
            task = self.market.tasks.get(tid)
            if task is not None:
                self.place_bid(task, now, bus, robots, peers)
        self._pending_bid = []

        for task in self.market.due_for_settlement(now):
            winner, cost = self.market.winner_of(task)
            if winner is None:
                self.market.settled.add(task.tid)
                continue
            if winner == self.rid:
                self.emit(msg_award(task.tid, winner, cost, self.rid, now),
                          bus, robots)
            elif self.market.apply_award(task.tid, winner, cost, now):
                self.on_award(task.tid, winner)

        live = [p.rid for p in peers]
        if self.market.i_am_lowest_live(live):
            for task in self.market.orphaned(live, now):
                if not self.market.reopen(task, now):
                    continue
                self.drop_local(task.tid)
                self.emit(msg_announce(task, self.rid, now), bus, robots)

        self.drive_execution(now, bus, robots)

    def place_bid(self, task, now, bus, robots, peers):
        if self.rid in task.excluded:
            return
        load = len(self.queue) + (1 if self.current else 0)
        if load >= self.params["max_queue"]:
            return
        to_pickup = self.planner.route_cost((self.x, self.y), task.pickup)
        leg = self.planner.route_cost(task.pickup, task.dropoff)
        busy = 0.0
        if self.current:
            cur = self.market.tasks.get(self.current)
            if cur is not None:
                target = cur.dropoff if self.phase == "TO_DROPOFF" else cur.pickup
                d = self.planner.route_cost((self.x, self.y), target)
                busy += d or 0.0
        congestion = sum(1 for p in peers
                         if math.hypot(p.x - task.pickup[0],
                                       p.y - task.pickup[1]) < 3.0)
        cost = bid_cost(to_pickup, leg, self.params["nominal_speed"],
                        queue_len=load, busy_remaining_m=busy,
                        congestion=congestion)
        if cost is None:
            return
        self.emit(msg_bid(task.tid, self.rid, cost, now), bus, robots)

    def drive_execution(self, now, bus, robots):
        if self.current is None:
            if self.queue:
                self.current = self.queue.pop(0)
                self.phase = "TO_PICKUP"
                self.send_goal(now, bus, robots)
            return
        if self.phase == "DWELL_PICKUP" and now >= self.dwell_until:
            self.phase = "TO_DROPOFF"
            self.send_goal(now, bus, robots)
        elif self.phase == "DWELL_DROPOFF" and now >= self.dwell_until:
            self.completed += 1
            self.emit(msg_done(self.current, self.rid, now), bus, robots)
            self.current, self.phase = None, "IDLE"

    def send_goal(self, now, bus, robots):
        task = self.market.tasks.get(self.current)
        if task is None:
            self.current, self.phase = None, "IDLE"
            return
        self.goal = task.pickup if self.phase == "TO_PICKUP" else task.dropoff
        self.reroutes = 0
        self.backoffs = 0
        self.hard_blockers.clear()
        self.pending_reroute = False
        self.nav_state = "PLANNING"
        self.state_since = now
        self.emit(msg_progress(task.tid, self.rid, self.phase, now),
                  bus, robots)

    def on_arrival(self, now):
        if self.phase == "TO_PICKUP":
            self.phase = "DWELL_PICKUP"
            self.dwell_until = now + self.params["dwell_s"]
        elif self.phase == "TO_DROPOFF":
            self.phase = "DWELL_DROPOFF"
            self.dwell_until = now + self.params["dwell_s"]
        self.nav_state = "IDLE"
        self.goal, self.path = None, []

    def release(self, now, bus, robots, reason):
        tid = self.current
        self.releases += 1
        self.emit(msg_release(tid, self.rid, reason, now), bus, robots)
        self.current, self.phase = None, "IDLE"
        self.nav_state, self.goal, self.path = "IDLE", None, []
        task = self.market.tasks.get(tid)
        if task is not None and task.state != DONE:
            self.emit(msg_announce(task, self.rid, now), bus, robots)

    # ------------------------------------------------------ navigator tick

    def dynamic_layer(self, now, peers):
        for rid in [r for r, exp in self.hard_blockers.items() if exp < now]:
            del self.hard_blockers[rid]
        out = []
        for p in peers:
            if p.speed < 0.05:
                self.peer_still_since.setdefault(p.rid, now)
            else:
                self.peer_still_since.pop(p.rid, None)
            since = self.peer_still_since.get(p.rid)
            hard = (p.rid in self.hard_blockers
                    or (since is not None and now - since > 4.0))
            out.append(DynamicObstacle(p.x, p.y, 0.85 if hard else 1.40,
                                       hard=hard, weight=4.0, rid=p.rid))
        return out

    def replan(self, now, peers):
        self.last_plan = now
        dynamic = self.dynamic_layer(now, peers)
        path = self.planner.plan((self.x, self.y), self.goal, dynamic=dynamic)
        if path is None and any(d.hard for d in dynamic):
            soft = [DynamicObstacle(d.x, d.y, 1.40, hard=False, weight=8.0,
                                    rid=d.rid) for d in dynamic]
            path = self.planner.plan((self.x, self.y), self.goal, dynamic=soft)
        if path is None:
            return False
        self.path = path
        self.follower.reset()
        return True

    def navigator_tick(self, now, dt, peers, bus, robots):
        if self.nav_state == "IDLE" or self.goal is None:
            self.cmd_v, self.cmd_w = 0.0, 0.0
            return

        if self.nav_state == "PLANNING":
            self.cmd_v, self.cmd_w = 0.0, 0.0
            self.nav_state = "FOLLOWING" if self.replan(now, peers) else "WAITING"
            self.state_since = now
            return

        if self.nav_state == "WAITING":
            self.cmd_v, self.cmd_w = 0.0, 0.0
            if now - self.state_since > self.params["wait_timeout_s"]:
                self.release(now, bus, robots, "unreachable")
                return
            if now - self.last_plan > 1.5 and self.replan(now, peers):
                self.reroutes = 0
                self.nav_state, self.state_since = "FOLLOWING", now
            return

        if self.nav_state == "BACKOFF":
            travelled = math.hypot(self.x - self.backoff_from[0],
                                   self.y - self.backoff_from[1])
            if travelled >= 1.20 or now - self.state_since > 10.0:
                self.cmd_v, self.cmd_w = 0.0, 0.0
                self.reroutes = 0
                self.nav_state, self.state_since = "PLANNING", now
            else:
                self.cmd_v, self.cmd_w = -0.18, 0.0
            return

        # FOLLOWING
        if len(self.path) < 2:
            self.nav_state = "PLANNING"
            return

        want = self.pending_reroute
        if not want and now - self.last_plan > self.params["replan_period_s"]:
            want = not self.planner.path_is_clear(
                self.path, self.dynamic_layer(now, peers),
                self.follower.index, look_ahead_m=6.0)
            self.last_plan = now

        if want:
            self.pending_reroute = False
            self.reroutes += 1
            if self.reroutes > self.params["max_reroutes"] \
                    or not self.replan(now, peers):
                self.begin_backoff(now, bus, robots)
                return

        result = self.follower.step(self.pose, abs(self.v), self.path,
                                    self.goal, self.cmd_v, dt)
        if result.arrived:
            self.cmd_v, self.cmd_w = 0.0, 0.0
            self.on_arrival(now)
            return
        self.cmd_v, self.cmd_w = result.v, result.w

    def begin_backoff(self, now, bus, robots):
        self.backoffs += 1
        if self.backoffs > self.params["max_backoffs"]:
            self.release(now, bus, robots, "blocked")
            return
        self.backoff_from = (self.x, self.y)
        self.nav_state, self.state_since = "BACKOFF", now

    # ------------------------------------------------------ collision tick

    def collision_tick(self, now, dt, peers):
        my_twist = (self.cmd_v * math.cos(self.yaw),
                    self.cmd_v * math.sin(self.yaw), self.cmd_w)
        if abs(self.cmd_v) < 1e-3 and abs(self.cmd_w) < 1e-3:
            my_twist = self.twist

        decision = self.resolver.resolve(
            now, self.pose, my_twist, abs(self.v), self.wait_broadcast,
            peers, engaged_prev=self.engaged)
        self.engaged = decision.engaged

        scale, status = decision.scale, decision.status
        translating = abs(self.cmd_v) > 1e-2
        rotating = abs(self.cmd_w) > 1e-3
        if (not translating and rotating and status != "EMERGENCY_STOP"
                and not self.resolver.rotation_is_safe(self.pose, peers)):
            scale, status = 0.0, "ROTATE_BLOCKED"

        # Same emit rules as the node: rotation in place is a separate
        # permission from driving, and linear/angular scale together so the
        # planned path geometry survives being traversed slowly.
        if status in ("EMERGENCY_STOP", "ROTATE_BLOCKED"):
            self.cmd_v, self.cmd_w = 0.0, 0.0
        elif not translating:
            pass
        else:
            self.cmd_v *= scale
            self.cmd_w *= scale

        held = (status.startswith("YIELD") or status.startswith("STOP")
                or status.startswith("SLOW")
                or status in ("EMERGENCY_STOP", "ROTATE_BLOCKED"))
        self.wait_s = (self.wait_s + dt) if held else max(0.0, self.wait_s - 2 * dt)
        self.hold_s = (self.hold_s + dt) if scale <= 0.0 else 0.0

        limit = {"head-on": 1.5, "blocked": 2.5}.get(decision.reason, 6.0)
        if (self.hold_s >= limit and decision.blocker
                and now - self.last_reroute >= 8.0):
            self.last_reroute = now
            self.hard_blockers[decision.blocker] = now + 12.0
            self.pending_reroute = True

        self.status = status
        self.scale = scale
        self.time_in_state[status] += dt

        if abs(self.v) < 0.02:
            self.stopped_for += dt
            self.longest_stop = max(self.longest_stop, self.stopped_for)
        else:
            self.stopped_for = 0.0


# ============================================================ the run

def read_yaml(path):
    import yaml
    with open(path, "r") as fh:
        return yaml.safe_load(fh) or {}


def run(args):
    rng = random.Random(args.seed)
    grid = load_map(os.path.join(ROOT, "config", "bcr_map.yaml"), downsample=3)
    cfg = read_yaml(os.path.join(ROOT, "config", "fleet_tasks.yaml"))
    stations = {k: (float(v[0]), float(v[1]))
                for k, v in cfg["stations"].items()}

    params = {
        "planning_clearance": 0.45, "max_linear": 0.40, "max_angular": 1.0,
        "max_accel": 0.35, "replan_period_s": 4.0, "max_reroutes": 3,
        "max_backoffs": 3, "wait_timeout_s": 12.0, "max_queue": 2,
        "nominal_speed": 0.35, "dwell_s": 2.0,
    }

    names = list(START_POSES)[:args.robots]
    robots = [SimRobot(rid, grid, params) for rid in names]
    for r in robots:
        r.now = 0.0
        r._pending_bid = []
    bus = Bus(rng, loss=args.loss)

    tasks = []
    for i, spec in enumerate(cfg["tasks"], 1):
        tasks.append(Task(spec.get("id", "T%02d" % i),
                          stations[spec["pickup"]], stations[spec["dropoff"]],
                          label=spec.get("label", ""), created=0.0))

    dt = 1.0 / 20.0
    steps = int(args.duration / dt)
    next_task = 0
    task_at = args.first_task
    min_gap = float("inf")
    min_gap_at = None
    collisions = 0
    completed_at = {}

    print("simulating %d robots for %.0f s, bus loss %.0f%%, seed %d"
          % (len(robots), args.duration, args.loss * 100, args.seed))
    print("map %s | %d tasks | dispatch every %.0f s\n"
          % (grid, len(tasks), args.task_interval))

    for step in range(steps):
        now = step * dt
        for r in robots:
            r.now = now

        # dispatcher
        if next_task < len(tasks) and now >= task_at:
            task = tasks[next_task]
            task.created = now
            for r in robots:
                t_copy = Task.from_dict(task.to_dict())
                if r.market.announce(t_copy, now):
                    r._pending_bid.append(task.tid)
            next_task += 1
            task_at = now + args.task_interval

        bus.flush(now)

        for r in robots:
            peers = bus.peers_for(r.rid)
            r.allocator_tick(now, bus, robots, peers)
            r.navigator_tick(now, dt, peers, bus, robots)
            r.collision_tick(now, dt, peers)

        for r in robots:
            r.integrate(dt)
            if int(now / 0.1) != int((now - dt) / 0.1):
                r.wait_broadcast = r.wait_s
            bus.broadcast_state(r.rid, r.sample(now))

        # ---- ground truth safety check, independent of what anyone believed
        for i in range(len(robots)):
            for j in range(i + 1, len(robots)):
                gap = capsule_clearance(robots[i].resolver.fp, robots[i].pose,
                                        robots[j].resolver.fp, robots[j].pose)
                if gap < min_gap:
                    min_gap, min_gap_at = gap, (now, robots[i].rid, robots[j].rid)
                if gap < 0.0:
                    collisions += 1

        for r in robots:
            for tid, t in r.market.tasks.items():
                if t.state == DONE and tid not in completed_at:
                    completed_at[tid] = now

    report(robots, tasks, completed_at, min_gap, min_gap_at, collisions, args)
    return failures(robots, tasks, completed_at, min_gap, collisions, args)


def report(robots, tasks, completed_at, min_gap, min_gap_at, collisions, args):
    print("=" * 78)
    print(" RESULT")
    print("=" * 78)
    print(" tasks issued            : %d" % len(tasks))
    print(" tasks completed         : %d" % len(completed_at))
    if completed_at:
        print(" last completion at      : %.1f s" % max(completed_at.values()))
    print(" hull overlaps (ticks)   : %d" % collisions)
    if min_gap_at:
        print(" closest hull gap ever   : %.3f m at t=%.1fs between %s and %s"
              % (min_gap, min_gap_at[0], min_gap_at[1], min_gap_at[2]))
    print()

    print(" %-6s %5s %9s %9s %8s %9s %9s" % ("ROBOT", "DONE", "DISTANCE",
                                             "LONGEST", "REROUTE", "BACKOFF",
                                             "RELEASE"))
    print(" %-6s %5s %9s %9s %8s %9s %9s" % ("", "", "(m)", "STOP (s)", "", "", ""))
    print(" " + "-" * 70)
    for r in robots:
        print(" %-6s %5d %9.1f %9.1f %8d %9d %9d"
              % (r.rid, r.completed, r.distance, r.longest_stop,
                 r.reroutes, r.backoffs, r.releases))
    print()

    print(" TIME BY AVOIDANCE STATE (this is the slide)")
    total = sum(sum(r.time_in_state.values()) for r in robots) or 1.0
    agg = defaultdict(float)
    for r in robots:
        for k, v in r.time_in_state.items():
            agg[k] += v
    for k in sorted(agg, key=lambda k: -agg[k]):
        print("   %-16s %7.1f s   %5.1f%%" % (k, agg[k], 100.0 * agg[k] / total))
    print()

    moving = sum(1 for r in robots if r.distance > 1.0)
    print(" robots that moved       : %d of %d" % (moving, len(robots)))
    print(" starvation check        : longest continuous stop %.1f s "
          "(threshold %.0f s)"
          % (max(r.longest_stop for r in robots), args.starvation))


def failures(robots, tasks, completed_at, min_gap, collisions, args):
    bad = []
    if collisions:
        bad.append("%d ticks of hull overlap (min gap %.3f m)"
                   % (collisions, min_gap))
    worst = max(r.longest_stop for r in robots)
    if worst > args.starvation:
        bad.append("a robot was stationary for %.1f s (threshold %.0f s)"
                   % (worst, args.starvation))
    idle = [r.rid for r in robots if r.distance < 1.0]
    if idle:
        bad.append("robots that never moved: %s" % ", ".join(idle))
    if len(completed_at) < len(tasks):
        bad.append("%d of %d tasks did not complete"
                   % (len(tasks) - len(completed_at), len(tasks)))

    print()
    if bad:
        print("FAILED:")
        for b in bad:
            print("  - " + b)
        return 1
    print("PASSED: no collisions, no starvation, every task completed.")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--robots", type=int, default=4)
    ap.add_argument("--duration", type=float, default=420.0)
    ap.add_argument("--task-interval", type=float, default=14.0)
    ap.add_argument("--first-task", type=float, default=3.0)
    ap.add_argument("--loss", type=float, default=0.0,
                    help="fraction of bus messages to drop (0.0-0.5)")
    ap.add_argument("--seed", type=int, default=20240901)
    ap.add_argument("--starvation", type=float, default=45.0,
                    help="fail if any robot is stationary this long")
    return run(ap.parse_args())


if __name__ == "__main__":
    sys.exit(main())
