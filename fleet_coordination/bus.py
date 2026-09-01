#!/usr/bin/env python3
"""The shared fleet bus: what robots tell each other, and how.

There is no broker and no fleet manager process. Every robot publishes its
own state onto a handful of global topics and subscribes to the same topics;
DDS discovery wires them together whenever a robot appears, in any order, on
any host in the domain. Kill any robot and the rest keep running -- the only
thing that happens is that its heartbeat goes stale and its peers notice.

Payloads are JSON inside ``std_msgs/String``. That is deliberate: it means
no custom .msg package to build, the bus is directly readable with
``ros2 topic echo``, and a dashboard or a log replay can consume the exact
bytes the robots exchanged.

Topics
------
``/fleet/state``         10 Hz heartbeat: pose, velocity, intent, wait credit
``/fleet/market``        auction traffic: ANNOUNCE / BID / AWARD / DONE / RELEASE
``/fleet/task_request``  new work entering the fleet (operator or dashboard)
``/fleet/events``        human-readable event stream for monitoring
"""

import json
import math

from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                       ReliabilityPolicy)

from std_msgs.msg import String

__all__ = [
    "FLEET_EVENT_TOPIC",
    "FLEET_MARKET_TOPIC",
    "FLEET_STATE_TOPIC",
    "FLEET_TASK_TOPIC",
    "FleetState",
    "PeerTable",
    "encode",
    "event_qos",
    "market_qos",
    "state_qos",
    "task_qos",
]

FLEET_STATE_TOPIC = "/fleet/state"
FLEET_MARKET_TOPIC = "/fleet/market"
FLEET_TASK_TOPIC = "/fleet/task_request"
FLEET_EVENT_TOPIC = "/fleet/events"


# ------------------------------------------------------------------- QoS
#
# State is a heartbeat: best effort, keep last. A dropped sample is
# replaced 100 ms later and stale data is worse than no data.
# Market traffic is transactional: reliable, and deep enough that a robot
# joining mid-auction does not silently miss bids.

def state_qos(depth=10):
    return QoSProfile(depth=depth,
                      reliability=ReliabilityPolicy.BEST_EFFORT,
                      history=HistoryPolicy.KEEP_LAST)


def market_qos(depth=50):
    return QoSProfile(depth=depth,
                      reliability=ReliabilityPolicy.RELIABLE,
                      history=HistoryPolicy.KEEP_LAST)


def task_qos(depth=50):
    """Task requests latch, so a robot that boots late still sees the backlog."""
    return QoSProfile(depth=depth,
                      reliability=ReliabilityPolicy.RELIABLE,
                      durability=DurabilityPolicy.TRANSIENT_LOCAL,
                      history=HistoryPolicy.KEEP_LAST)


def event_qos(depth=100):
    return QoSProfile(depth=depth,
                      reliability=ReliabilityPolicy.RELIABLE,
                      history=HistoryPolicy.KEEP_LAST)


def encode(payload):
    """Dict -> ``std_msgs/String``. Separators keep the wire compact."""
    return String(data=json.dumps(payload, separators=(",", ":")))


# ------------------------------------------------------------ state item

class FleetState:
    """One robot's broadcast, as seen by its peers.

    Everything a peer needs to reason about this robot without asking it a
    question. In particular ``wait`` (accumulated yield time) is broadcast
    rather than inferred, because right-of-way arbitration has to be
    computable identically on both sides of a pair -- see
    :mod:`fleet_coordination.arbitration`.
    """

    __slots__ = ("rid", "stamp", "x", "y", "yaw", "vx", "vy", "omega",
                 "speed", "priority", "wait", "state", "task_id", "phase",
                 "goal", "seq")

    def __init__(self, **kw):
        self.rid = kw["id"]
        self.stamp = float(kw.get("t", 0.0))
        self.x = float(kw.get("x", 0.0))
        self.y = float(kw.get("y", 0.0))
        self.yaw = float(kw.get("yaw", 0.0))
        self.vx = float(kw.get("vx", 0.0))
        self.vy = float(kw.get("vy", 0.0))
        self.omega = float(kw.get("w", 0.0))
        self.speed = float(kw.get("v", math.hypot(self.vx, self.vy)))
        self.priority = float(kw.get("prio", 1.0))
        self.wait = float(kw.get("wait", 0.0))
        self.state = kw.get("state", "UNKNOWN")
        self.task_id = kw.get("task")
        self.phase = kw.get("phase", "IDLE")
        g = kw.get("goal")
        self.goal = (float(g[0]), float(g[1])) if g else None
        self.seq = int(kw.get("seq", 0))

    @property
    def pose(self):
        return (self.x, self.y, self.yaw)

    @property
    def twist(self):
        return (self.vx, self.vy, self.omega)

    @staticmethod
    def payload(rid, stamp, pose, twist, speed, priority, wait, state,
                task_id=None, phase="IDLE", goal=None, seq=0):
        return {
            "id": rid,
            "t": round(stamp, 3),
            "x": round(pose[0], 3),
            "y": round(pose[1], 3),
            "yaw": round(pose[2], 4),
            "vx": round(twist[0], 3),
            "vy": round(twist[1], 3),
            "w": round(twist[2], 3),
            "v": round(speed, 3),
            "prio": priority,
            "wait": round(wait, 2),
            "state": state,
            "task": task_id,
            "phase": phase,
            "goal": [round(goal[0], 2), round(goal[1], 2)] if goal else None,
            "seq": seq,
        }

    def __repr__(self):
        return (f"<{self.rid} @({self.x:.2f},{self.y:.2f}) v={self.speed:.2f} "
                f"{self.state} task={self.task_id}>")


# ------------------------------------------------------------ peer table

class PeerTable:
    """Peers heard recently, with staleness handled in one place.

    A peer that stops broadcasting is not "stopped" -- it is *unknown*, and
    the two demand different responses. Unknown peers are dropped from
    collision reasoning (we have no velocity to predict with) but their
    disappearance is reported to the task layer, which reassigns whatever
    they were carrying.
    """

    def __init__(self, timeout_s=1.5):
        self.timeout = float(timeout_s)
        self._peers = {}
        self._last_seen = {}

    def update(self, payload, now):
        rid = payload.get("id")
        if rid is None:
            return None
        prev = self._peers.get(rid)
        if prev is not None and payload.get("seq", 0) < prev.seq:
            return None                       # out-of-order sample, ignore
        st = FleetState(**payload)
        self._peers[rid] = st
        self._last_seen[rid] = now
        return st

    def prune(self, now):
        """Drop peers whose heartbeat went stale. Returns the dropped ids."""
        dead = [rid for rid, t in self._last_seen.items()
                if now - t > self.timeout]
        for rid in dead:
            self._peers.pop(rid, None)
            self._last_seen.pop(rid, None)
        return dead

    def get(self, rid):
        return self._peers.get(rid)

    def ids(self):
        return list(self._peers.keys())

    def values(self):
        return list(self._peers.values())

    def items(self):
        return list(self._peers.items())

    def __len__(self):
        return len(self._peers)

    def __contains__(self, rid):
        return rid in self._peers
