#!/usr/bin/env python3
"""Inject work into the fleet.

This is not a fleet manager. It has no idea which robot will do anything --
it publishes transport jobs onto ``/fleet/task_request`` and the robots
auction them out among themselves. Kill it after it has published and the
fleet carries on; start a second one and the tasks simply add.

The topic is TRANSIENT_LOCAL, so a robot that boots after the dispatcher has
run still receives the backlog rather than missing it. That is what lets you
stage the demo in whatever order is convenient on the day.

Usage::

    # the scripted job list from config/fleet_tasks.yaml
    ros2 run bcr_bot fleet_task_dispatcher.py --ros-args \\
        -p tasks_file:=/path/to/fleet_tasks.yaml

    # continuous random traffic between the named stations
    ros2 run bcr_bot fleet_task_dispatcher.py --ros-args \\
        -p tasks_file:=/path/to/fleet_tasks.yaml -p mode:=random \\
        -p interval_s:=12.0
"""

import os
import random

import rclpy
from rclpy.node import Node

from std_msgs.msg import String

from fleet_coordination.bus import FLEET_TASK_TOPIC, encode, task_qos
from fleet_coordination.tasks import Task


def read_yaml(path):
    try:
        import yaml
        with open(path, "r") as fh:
            return yaml.safe_load(fh) or {}
    except ImportError:
        raise RuntimeError("PyYAML is required to read %s" % path)


class FleetTaskDispatcher(Node):

    def __init__(self):
        super().__init__("fleet_task_dispatcher")

        self.declare_parameters("", [
            ("tasks_file", ""),
            ("mode", "scripted"),        # scripted | random
            ("interval_s", 6.0),
            ("start_delay_s", 6.0),
            ("random_count", 0),         # 0 = unlimited
            ("seed", 20240901),
        ])

        g = self.get_parameter
        self.mode = g("mode").value
        self.interval = float(g("interval_s").value)
        self.random_count = int(g("random_count").value)
        self.rng = random.Random(int(g("seed").value))

        path = os.path.expanduser(g("tasks_file").value or "")
        if not path or not os.path.exists(path):
            raise SystemExit("fleet_task_dispatcher: tasks_file %r not found"
                             % path)
        cfg = read_yaml(path)
        self.stations = {k: (float(v[0]), float(v[1]))
                         for k, v in (cfg.get("stations") or {}).items()}
        self.scripted = cfg.get("tasks") or []

        if not self.stations:
            raise SystemExit("fleet_task_dispatcher: no stations in %s" % path)

        self.pub = self.create_publisher(String, FLEET_TASK_TOPIC, task_qos())
        self.issued = 0
        self.index = 0

        self.get_logger().info(
            "dispatcher up | %d stations | mode=%s | %d scripted tasks"
            % (len(self.stations), self.mode, len(self.scripted)))

        self.timer = self.create_timer(float(g("start_delay_s").value),
                                       self.first_shot)

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def first_shot(self):
        self.timer.cancel()
        self.timer = self.create_timer(self.interval, self.publish_next)
        self.publish_next()

    # ------------------------------------------------------------------

    def resolve(self, name):
        if name in self.stations:
            return self.stations[name]
        raise KeyError("unknown station %r" % name)

    def publish_next(self):
        if self.mode == "random":
            task = self.random_task()
        else:
            task = self.scripted_task()

        if task is None:
            self.get_logger().info("dispatcher: %d tasks issued, idling"
                                   % self.issued)
            self.timer.cancel()
            return

        self.pub.publish(encode(task.to_dict()))
        self.issued += 1
        self.get_logger().info(
            "issued %s [%s]  pickup=(%.2f,%.2f) dropoff=(%.2f,%.2f)"
            % (task.tid, task.label, task.pickup[0], task.pickup[1],
               task.dropoff[0], task.dropoff[1]))

    def scripted_task(self):
        if self.index >= len(self.scripted):
            return None
        spec = self.scripted[self.index]
        self.index += 1
        tid = spec.get("id") or ("T%02d" % self.index)
        pickup = (self.resolve(spec["pickup"]) if isinstance(spec["pickup"], str)
                  else tuple(spec["pickup"]))
        dropoff = (self.resolve(spec["dropoff"]) if isinstance(spec["dropoff"], str)
                   else tuple(spec["dropoff"]))
        return Task(tid, pickup, dropoff,
                    label=spec.get("label", "%s->%s" % (spec["pickup"],
                                                        spec["dropoff"])),
                    priority=float(spec.get("priority", 1.0)),
                    created=self.now())

    def random_task(self):
        if self.random_count and self.issued >= self.random_count:
            return None
        names = list(self.stations)
        a = self.rng.choice(names)
        b = self.rng.choice([n for n in names if n != a])
        self.issued_id = "R%03d" % (self.issued + 1)
        return Task(self.issued_id, self.stations[a], self.stations[b],
                    label="%s->%s" % (a, b), created=self.now())


def main():
    rclpy.init()
    try:
        node = FleetTaskDispatcher()
    except SystemExit as exc:
        print(exc)
        rclpy.shutdown()
        return
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
