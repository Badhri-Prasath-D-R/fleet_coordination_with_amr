#!/usr/bin/env python3
"""Read-only view of the whole fleet, for demos and for post-mortems.

This node subscribes and never publishes a command. It exists to make the
decentralized part *visible*: everything on screen was reconstructed from
the same broadcast topics the robots use to coordinate with each other, by a
process that has no privileged access to any of them. Kill it and nothing
changes; start three of them and nothing changes either.

Two outputs:

* a table refreshed in place, showing each robot pose, speed, avoidance
  state, wait credit and current task;
* an append-only event log, so a run can be replayed and quoted afterwards.
"""

import json
import os
import sys
from datetime import datetime

import rclpy
from rclpy.node import Node

from std_msgs.msg import String

from fleet_coordination.bus import (FLEET_EVENT_TOPIC, FLEET_MARKET_TOPIC,
                                    FLEET_STATE_TOPIC, FLEET_TASK_TOPIC,
                                    PeerTable, event_qos, market_qos,
                                    state_qos, task_qos)

STATE_ORDER = ("CLEAR", "PROCEED_ROW", "YIELD_SLOW", "SLOW_BLOCKED",
               "YIELD_STOP", "STOP_BLOCKED", "YIELD_HEADON", "ROTATE_BLOCKED",
               "LASER_STOP", "EMERGENCY_STOP")


class FleetMonitor(Node):

    def __init__(self):
        super().__init__("fleet_monitor")

        self.declare_parameters("", [
            ("refresh_hz", 2.0),
            ("peer_timeout_s", 3.0),
            ("event_log", ""),
            ("clear_screen", True),
        ])

        g = self.get_parameter
        self.clear_screen = bool(g("clear_screen").value)
        self.robots = PeerTable(timeout_s=float(g("peer_timeout_s").value))
        self.events = []
        self.tasks = {}
        self.owners = {}
        self.counters = {"announced": 0, "done": 0, "released": 0,
                         "reopened": 0, "preempted": 0}
        self.state_seconds = {}
        self.last_tick = None

        path = g("event_log").value
        self.fh = None
        if path:
            path = os.path.expanduser(path)
            try:
                os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
                self.fh = open(path, "a", buffering=1)
            except OSError as exc:
                print("[monitor] cannot open %s: %s" % (path, exc))

        self.create_subscription(String, FLEET_STATE_TOPIC, self.on_state,
                                 state_qos())
        self.create_subscription(String, FLEET_EVENT_TOPIC, self.on_event,
                                 event_qos())
        self.create_subscription(String, FLEET_MARKET_TOPIC, self.on_market,
                                 market_qos())
        self.create_subscription(String, FLEET_TASK_TOPIC, self.on_task,
                                 task_qos())

        self.create_timer(1.0 / max(0.2, float(g("refresh_hz").value)),
                          self.render)

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    # ---------------------------------------------------------------- I/O

    def on_state(self, msg):
        try:
            self.robots.update(json.loads(msg.data), self.now())
        except (ValueError, TypeError):
            pass

    def on_event(self, msg):
        try:
            d = json.loads(msg.data)
        except (ValueError, TypeError):
            return
        kind = d.get("event", "")
        for key in self.counters:
            if kind.endswith(key):
                self.counters[key] += 1
        line = "[%s] %-16s %-6s %s" % (
            datetime.now().strftime("%H:%M:%S"), kind, d.get("from", "?"),
            " ".join("%s=%s" % (k, v) for k, v in sorted(d.items())
                     if k not in ("event", "from", "t")))
        self.events.append(line)
        del self.events[:-14]
        if self.fh:
            self.fh.write(line + "\n")

    def on_market(self, msg):
        try:
            d = json.loads(msg.data)
        except (ValueError, TypeError):
            return
        tid = d.get("tid") or (d.get("task") or {}).get("tid")
        if not tid:
            return
        kind = d.get("type")
        if kind == "AWARD":
            self.owners[tid] = d.get("winner")
        elif kind == "DONE":
            self.owners[tid] = "done"
        elif kind == "RELEASE":
            self.owners[tid] = "released"

    def on_task(self, msg):
        try:
            d = json.loads(msg.data)
        except (ValueError, TypeError):
            return
        if d.get("tid"):
            self.tasks[d["tid"]] = d

    # ------------------------------------------------------------- render

    def render(self):
        now = self.now()
        self.robots.prune(now)

        if self.last_tick is not None:
            dt = now - self.last_tick
            for r in self.robots.values():
                self.state_seconds[r.state] = \
                    self.state_seconds.get(r.state, 0.0) + dt
        self.last_tick = now

        rows = sorted(self.robots.values(), key=lambda r: r.rid)
        out = []
        out.append("=" * 100)
        out.append(" FLEET MONITOR   sim=%8.1fs   robots=%d   "
                   "tasks: %d announced / %d done / %d released / %d reopened"
                   % (now, len(rows), self.counters["announced"],
                      self.counters["done"], self.counters["released"],
                      self.counters["reopened"]))
        out.append("=" * 100)
        out.append(" %-6s %-16s %6s %6s %-15s %6s  %-8s %-12s %s"
                   % ("ROBOT", "POSITION", "SPEED", "YAW", "AVOIDANCE",
                      "WAIT", "TASK", "PHASE", "GOAL"))
        out.append("-" * 100)
        for r in rows:
            out.append(" %-6s (%6.2f,%6.2f) %6.2f %6.2f %-15s %6.1f  %-8s "
                       "%-12s %s"
                       % (r.rid, r.x, r.y, r.speed, r.yaw, r.state, r.wait,
                          r.task_id or "-", r.phase,
                          "(%.1f,%.1f)" % r.goal if r.goal else "-"))
        if not rows:
            out.append("   (no robots broadcasting on %s)" % FLEET_STATE_TOPIC)

        out.append("-" * 100)
        busy = sum(1 for r in rows if r.state != "CLEAR")
        out.append(" avoidance active on %d/%d robots   |   time by state: %s"
                   % (busy, len(rows), self.state_summary()))
        out.append("-" * 100)
        out.append(" RECENT EVENTS")
        for line in self.events[-12:]:
            out.append("   " + line)
        if not self.events:
            out.append("   (none yet)")

        text = "\n".join(out)
        if self.clear_screen:
            sys.stdout.write("\033[2J\033[H")
        sys.stdout.write(text + "\n")
        sys.stdout.flush()

    def state_summary(self):
        """Where the fleet actually spends its time -- the number for the slide."""
        total = sum(self.state_seconds.values()) or 1.0
        parts = []
        for key in STATE_ORDER:
            secs = self.state_seconds.get(key, 0.0)
            if secs > 0.05:
                parts.append("%s %.0f%%" % (key, 100.0 * secs / total))
        return "  ".join(parts) if parts else "n/a"

    def destroy_node(self):
        if self.fh:
            self.fh.close()
        super().destroy_node()


def main():
    rclpy.init()
    node = FleetMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
