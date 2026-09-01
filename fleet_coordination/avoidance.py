#!/usr/bin/env python3
"""The conflict resolver: predict, arbitrate, govern.

Deliberately free of ROS. Everything the collision node decides about peers
happens in :meth:`ConflictResolver.resolve`, which is a pure function of the
robot own pose and intended motion plus what its peers last broadcast. The
node around it does the plumbing -- odometry, the laser floor, publishing --
and the offline simulator drives this same class directly, so the behaviour
that gets validated is the behaviour that ships.

The pipeline for each peer, cheapest test first:

1. **Hull clearance now.** Below ``emergency_clearance`` nothing else matters
   and nobody has right of way.
2. **Point-mass CPA.** Cheap, and its sign alone answers the question raw
   distance cannot: are these two converging at all? A negative time to
   closest approach means they are separating, which retires the entire
   class of false positives that made up half the original braking.
3. **Swept footprint.** Roll both chassis forward across the horizon and
   find the first moment their hulls come within the safety margin.
   Sampling the sweep rather than testing only the CPA instant matters for
   turning robots and long chassis, where least *hull* clearance and least
   *centre* separation happen at different times.
4. **Arbitration.** Aged priority, latched for the encounter -- see
   :mod:`fleet_coordination.arbitration`.
5. **Governor.** A continuous speed scale, never a binary stop.
"""

import math

from fleet_coordination.arbitration import RightOfWay
from fleet_coordination.control import speed_governor
from fleet_coordination.geometry import (Footprint, angle_diff,
                                         capsule_clearance,
                                         closest_point_of_approach,
                                         sweep_clearance)

__all__ = ["ConflictReport", "Decision", "PeerView", "ConflictResolver"]

# Status values, worst last. Anything from YIELD onward is a hold caused by
# a peer, and earns wait credit that buys right of way back later.
CLEAR = "CLEAR"
PROCEED_ROW = "PROCEED_ROW"
YIELD_SLOW = "YIELD_SLOW"
SLOW_BLOCKED = "SLOW_BLOCKED"
YIELD_STOP = "YIELD_STOP"
STOP_BLOCKED = "STOP_BLOCKED"
YIELD_HEADON = "YIELD_HEADON"
EMERGENCY_STOP = "EMERGENCY_STOP"


class PeerView:
    """What this robot knows about one peer. Mirrors a /fleet/state sample."""

    __slots__ = ("rid", "pose", "twist", "speed", "priority", "wait", "state")

    def __init__(self, rid, pose, twist, speed, priority, wait, state="CLEAR"):
        self.rid = rid
        self.pose = pose            # (x, y, yaw)
        self.twist = twist          # world frame (vx, vy, omega)
        self.speed = speed
        self.priority = priority
        self.wait = wait
        self.state = state

    @property
    def x(self):
        return self.pose[0]

    @property
    def y(self):
        return self.pose[1]


class ConflictReport:
    """One pair, one tick -- everything needed to log or draw the encounter."""

    __slots__ = ("rid", "peer", "t_cpa", "d_cpa", "t_hit", "gap_now",
                 "gap_min", "kind", "i_have_row", "status", "scale")

    def __init__(self, rid, peer, t_cpa, d_cpa, t_hit, gap_now, gap_min,
                 kind, i_have_row, status, scale):
        self.rid = rid
        self.peer = peer
        self.t_cpa = t_cpa
        self.d_cpa = d_cpa
        self.t_hit = t_hit
        self.gap_now = gap_now
        self.gap_min = gap_min
        self.kind = kind
        self.i_have_row = i_have_row
        self.status = status
        self.scale = scale


class Decision:
    """The resolver verdict for one tick."""

    __slots__ = ("scale", "status", "blocker", "reason", "reports", "engaged")

    def __init__(self, scale, status, blocker, reason, reports, engaged):
        self.scale = scale
        self.status = status
        self.blocker = blocker
        self.reason = reason
        self.reports = reports
        self.engaged = engaged

    def is_hold(self):
        return self.status != CLEAR and self.status != PROCEED_ROW


class ConflictResolver:

    def __init__(self, rid, base_priority, footprint_length=0.90,
                 footprint_width=0.64, safety_margin=0.25, hysteresis=0.20,
                 emergency_clearance=0.15, horizon=8.0, sweep_step=0.25,
                 brake_ttc=2.5, stop_ttc=1.0, min_scale=0.15, row_scale=0.85,
                 headon_deg=135.0, overtake_deg=45.0, still_speed=0.05,
                 aging_rate=0.25, deadband=0.05, max_wait_credit=30.0,
                 latch_release_s=1.0):
        self.rid = rid
        self.fp = Footprint(footprint_length, footprint_width)
        self.circumscribed = self.fp.half_len + self.fp.radius
        self.margin = safety_margin
        self.hysteresis = hysteresis
        self.emergency = emergency_clearance
        self.horizon = horizon
        self.sweep_step = sweep_step
        self.brake_ttc = brake_ttc
        self.stop_ttc = stop_ttc
        self.min_scale = min_scale
        self.row_scale = row_scale
        self.headon_rad = math.radians(headon_deg)
        self.overtake_rad = math.radians(overtake_deg)
        self.still_speed = still_speed
        self.row = RightOfWay(rid, base_priority, aging_rate=aging_rate,
                              deadband=deadband,
                              max_wait_credit=max_wait_credit,
                              latch_release_s=latch_release_s)

    # ------------------------------------------------------ classification

    def classify(self, my_pose, my_speed, my_wants_to_move, peer):
        """head-on / crossing / overtaking / blocked.

        The distinction drives the *style* of resolution. A crossing conflict
        is resolved beautifully by one robot arriving a second later, which
        is exactly what a speed governor produces. A head-on conflict is not:
        slowing in a 1.8 m aisle only decides who reaches the stalemate
        first, so it has to escalate to a reroute. And a peer that is parked
        is not in a conflict with us at all -- it is an obstacle, and the
        answer to an obstacle is to go around it, not to queue behind it.
        """
        if peer.speed < self.still_speed:
            return "crossing" if peer.state.startswith("YIELD") else "blocked"
        if my_speed < self.still_speed and not my_wants_to_move:
            return "crossing"
        d = angle_diff(my_pose[2], peer.pose[2])
        if d > self.headon_rad:
            return "head-on"
        if d < self.overtake_rad:
            return "overtaking"
        return "crossing"

    def rotation_is_safe(self, my_pose, peers, margin=0.10):
        """May we turn on the spot while translation is forbidden?

        A capsule pivoting in place sweeps a disc of its circumscribed
        radius, so this is not free -- but it is often available when driving
        is not, and it is what lets a stopped robot turn onto the detour its
        planner just handed it instead of sitting there.
        """
        reach = 2.0 * self.circumscribed + margin
        for peer in peers:
            if math.hypot(peer.x - my_pose[0], peer.y - my_pose[1]) < reach:
                return False
        return True

    # ------------------------------------------------------------- resolve

    def resolve(self, now, my_pose, my_twist, my_speed, my_wait_broadcast,
                peers, engaged_prev=()):
        """Decide this tick speed scale against every known peer.

        ``my_twist`` should be the *desired* world-frame velocity -- what the
        robot would do if nothing were in the way -- not the measured one.
        Predicting from measured velocity is self-defeating for a robot that
        is already yielding: once it stops, the prediction says the conflict
        is gone, so it releases, accelerates, re-detects and brakes again.
        That limit cycle is what the 4.5 s lockstep in the 09:18 log looks
        like from the inside.
        """
        self.row.expire(now)

        scale = 1.0
        status = CLEAR
        blocker = None
        reason = None
        engaged = set()
        reports = []
        wants_to_move = math.hypot(my_twist[0], my_twist[1]) > 1e-3

        for peer in peers:
            # 0. reachability bound. Neither robot can close more than
            # (|v_a| + |v_b|) * horizon of ground, so a peer further away
            # than that plus both hulls plus the margin cannot possibly be
            # a conflict inside the horizon. This is an exact upper bound,
            # not a heuristic, and it keeps the sweep off the hot path for
            # the overwhelming majority of peers on a busy floor.
            centre_dist = math.hypot(peer.pose[0] - my_pose[0],
                                     peer.pose[1] - my_pose[1])
            closing = (math.hypot(my_twist[0], my_twist[1])
                       + math.hypot(peer.twist[0], peer.twist[1]))
            reach = (closing * self.horizon + 2.0 * self.circumscribed
                     + self.margin + self.hysteresis)
            if centre_dist > reach:
                continue

            gap_now = capsule_clearance(self.fp, my_pose, self.fp, peer.pose)

            # 1. unconditional floor -- right of way does not apply here
            if gap_now < self.emergency:
                engaged.add(peer.rid)
                if status != EMERGENCY_STOP:
                    scale, status, blocker = 0.0, EMERGENCY_STOP, peer.rid
                    reason = "emergency"
                reports.append(ConflictReport(
                    self.rid, peer, 0.0, gap_now, 0.0, gap_now, gap_now,
                    "emergency", False, EMERGENCY_STOP, 0.0))
                continue

            # 2. cheap converging test
            t_cpa, d_cpa = closest_point_of_approach(
                (my_pose[0], my_pose[1]), (my_twist[0], my_twist[1]),
                (peer.pose[0], peer.pose[1]), (peer.twist[0], peer.twist[1]))

            # 3. swept footprint, with hysteresis once already engaged
            latched = peer.rid in engaged_prev
            threshold = self.margin + (self.hysteresis if latched else 0.0)

            # Under pure translation, a pair whose closest approach is
            # already behind them is at its minimum right now, so if that
            # minimum clears the margin there is nothing to sweep for. This
            # is the test that retires the false positives outright: in the
            # 09:16 encounter the last 40 samples were taken while the two
            # robots drove apart, and every one of them braked.
            spinning = (abs(my_twist[2]) > 1e-2 or abs(peer.twist[2]) > 1e-2)
            if t_cpa < 0.0 and not spinning and gap_now >= threshold:
                continue

            t_hit, gap_min, _t_min = sweep_clearance(
                self.fp, my_pose, my_twist,
                self.fp, peer.pose, peer.twist,
                self.horizon, self.sweep_step, threshold)

            if t_hit is None:
                continue

            engaged.add(peer.rid)

            # 4. arbitration
            kind = self.classify(my_pose, my_speed, wants_to_move, peer)
            i_go = self.row.decide(now, peer.rid, my_wait_broadcast,
                                   peer.priority, peer.wait)
            peer_yielding = peer.state.startswith("YIELD")

            # 5. governor
            if i_go and (peer.speed >= self.still_speed or peer_yielding):
                # The peer is cooperating: hold our line, shave a little
                # speed so the pass is smooth rather than a swerve.
                s, st, why = self.row_scale, PROCEED_ROW, None
            elif i_go:
                # Right of way over something that is not going to move is
                # meaningless. Treat it as the obstacle it is.
                s = speed_governor(t_hit, self.brake_ttc, self.stop_ttc,
                                   self.min_scale)
                st = SLOW_BLOCKED if s > 0.0 else STOP_BLOCKED
                why = "blocked"
            elif kind == "head-on":
                s, st, why = 0.0, YIELD_HEADON, "head-on"
            elif kind == "blocked":
                s = speed_governor(t_hit, self.brake_ttc, self.stop_ttc,
                                   self.min_scale)
                st = SLOW_BLOCKED if s > 0.0 else STOP_BLOCKED
                why = "blocked"
            else:
                s = speed_governor(t_hit, self.brake_ttc, self.stop_ttc,
                                   self.min_scale)
                st = YIELD_SLOW if s > 0.0 else YIELD_STOP
                why = None

            reports.append(ConflictReport(
                self.rid, peer, t_cpa, d_cpa, t_hit, gap_now, gap_min,
                kind, i_go, st, s))

            if s < scale:
                scale, status, blocker, reason = s, st, peer.rid, why

        return Decision(scale, status, blocker, reason, reports, engaged)
