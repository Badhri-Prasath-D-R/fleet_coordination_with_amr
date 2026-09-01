#!/usr/bin/env python3
"""Structured fleet logging.

The single most expensive gap in the original logs was that they recorded
positions and nothing else. ``dist=1.20m`` is the same string whether the
pair is closing head-on at 0.5 m/s or driving apart, so after the fact there
is no way to tell a genuine near-miss from a robot that was parked -- and no
way to answer "why did amr2 stop" when someone asks.

Every line here therefore carries velocity, heading, current goal and task,
and the simulation timestamp alongside the wall clock. The ``[PROXIMITY]``
line keeps its original leading fields so old tooling still reads it, and
appends the new ones; :mod:`scripts/replay_proximity_log.py` parses field by
field rather than with one brittle pattern, so both generations of log
replay correctly.
"""

import os
from datetime import datetime

__all__ = ["FleetLogger", "fmt_xy"]


def fmt_xy(p, nd=2):
    if p is None:
        return "none"
    return "(%.*f,%.*f)" % (nd, p[0], nd, p[1])


class FleetLogger:
    """Writes to stdout and, optionally, to a per-robot file.

    ``sim_clock`` is a zero-argument callable returning simulation seconds.
    Under Gazebo the wall clock and the sim clock diverge whenever the
    simulation is not running at real time, and it is the sim clock that the
    kinematics were computed against -- so both go on every line.
    """

    def __init__(self, robot_id, path=None, sim_clock=None, echo=True,
                 ros_logger=None):
        self.rid = robot_id
        self.echo = echo
        self.ros_logger = ros_logger
        self.sim_clock = sim_clock or (lambda: 0.0)
        self.fh = None
        if path:
            path = os.path.expanduser(path)
            directory = os.path.dirname(path)
            if directory:
                try:
                    os.makedirs(directory, exist_ok=True)
                except OSError:
                    pass
            try:
                self.fh = open(path, "a", buffering=1)
            except OSError as exc:
                print("[flog] cannot open %s: %s" % (path, exc), flush=True)
        self.path = path

    # ------------------------------------------------------------- core

    def line(self, tag, text):
        stamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        out = "[%s] [%s] sim=%.2f %s" % (stamp, tag, self.sim_clock(), text)
        if self.echo:
            print(out, flush=True)
        if self.fh:
            try:
                self.fh.write(out + "\n")
            except OSError:
                pass
        return out

    # ---------------------------------------------------------- records

    def state(self, pose, twist, speed, status, goal=None, task=None,
              scale=1.0):
        self.line("STATE",
                  "%s pos=%s yaw=%.3f vel=(%.3f,%.3f) w=%.3f v=%.3f "
                  "state=%s scale=%.2f goal=%s task=%s"
                  % (self.rid, fmt_xy(pose), pose[2], twist[0], twist[1],
                     twist[2], speed, status, scale, fmt_xy(goal), task or "none"))

    def proximity(self, peer_id, dist, gap, me, peer):
        """One pair, one sample.

        ``me`` and ``peer`` are dicts with keys pos, yaw, vel, prio, goal, task.
        The leading ``dist=`` field is preserved verbatim from the original
        format so that logs from before and after this change are directly
        comparable in the replay scorer.
        """
        self.line(
            "PROXIMITY",
            "%s <-> %s | dist=%.2fm | gap=%.2fm | "
            "%s: pos=%s yaw=%.3f vel=(%.3f,%.3f) prio=%s goal=%s task=%s | "
            "%s: pos=%s yaw=%.3f vel=(%.3f,%.3f) prio=%s goal=%s task=%s"
            % (self.rid, peer_id, dist, gap,
               self.rid, fmt_xy(me["pos"]), me["yaw"], me["vel"][0],
               me["vel"][1], me["prio"], fmt_xy(me.get("goal")),
               me.get("task") or "none",
               peer_id, fmt_xy(peer["pos"]), peer["yaw"], peer["vel"][0],
               peer["vel"][1], peer["prio"], fmt_xy(peer.get("goal")),
               peer.get("task") or "none"))

    def conflict(self, peer_id, t_cpa, d_cpa, t_hit, gap_min, kind, row,
                 action, scale):
        self.line("CONFLICT",
                  "%s <-> %s | t_cpa=%.2fs d_cpa=%.2fm t_hit=%s gap_min=%.2fm "
                  "class=%s row=%s action=%s scale=%.2f"
                  % (self.rid, peer_id, t_cpa, d_cpa,
                     "none" if t_hit is None else "%.2fs" % t_hit,
                     gap_min, kind, row, action, scale))

    def reroute(self, reason, peer_id=None, old_len=None, new_len=None,
                attempt=0):
        self.line("REROUTE",
                  "%s reason=%s peer=%s attempt=%d old_len=%s new_len=%s"
                  % (self.rid, reason, peer_id or "none", attempt,
                     "none" if old_len is None else "%.2fm" % old_len,
                     "none" if new_len is None else "%.2fm" % new_len))

    def task(self, verb, tid, detail=""):
        self.line("TASK", "%s %s %s %s" % (self.rid, verb, tid, detail))

    def event(self, text):
        self.line("EVENT", "%s %s" % (self.rid, text))

    def warn(self, text):
        out = self.line("WARN", "%s %s" % (self.rid, text))
        if self.ros_logger is not None:
            self.ros_logger.warn(text)
        return out

    def close(self):
        if self.fh:
            try:
                self.fh.close()
            except OSError:
                pass
            self.fh = None
