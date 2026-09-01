#!/usr/bin/env python3
"""Check the fleet configuration against the map before you run anything.

Catches the class of failure that is miserable to debug live: a station or a
spawn pose that sits a few centimetres inside a shelf leg, so the robot
either cannot be spawned there or nobody can bid on the task. It needs no
ROS, no Gazebo and no build -- run it straight from the package root:

    python3 scripts/verify_fleet_config.py

Exits non-zero if anything is unroutable, so it can also go in CI.
"""

import itertools
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from fleet_coordination.gridmap import load_map          # noqa: E402
from fleet_coordination.planner import Planner           # noqa: E402

MAP_YAML = os.path.join(ROOT, "config", "bcr_map.yaml")
TASKS_YAML = os.path.join(ROOT, "config", "fleet_tasks.yaml")

PLANNING_CLEARANCE = 0.45      # circumscribed radius of the bcr_bot chassis
SPAWN_CLEARANCE = 0.80         # a spawn wants more room than a transit cell

# Kept in step with launch/multi_robot_warehouse.launch.py.
SPAWNS = {
    "amr1": (-4.50, -7.75),
    "amr2": (-5.00, -3.50),
    "amr3": (-2.00, -0.25),
    "amr4": (-4.75, 1.25),
}


def read_yaml(path):
    try:
        import yaml
    except ImportError:
        sys.exit("PyYAML is required: pip3 install pyyaml")
    with open(path, "r") as fh:
        return yaml.safe_load(fh) or {}


def main():
    grid = load_map(MAP_YAML, downsample=3)
    planner = Planner(grid, min_clearance=PLANNING_CLEARANCE)
    cfg = read_yaml(TASKS_YAML)
    stations = {k: (float(v[0]), float(v[1]))
                for k, v in (cfg.get("stations") or {}).items()}

    print("map      : %s" % grid)
    print("           %s" % grid.stats())
    print("clearance: transit >= %.2f m, spawn >= %.2f m"
          % (PLANNING_CLEARANCE, SPAWN_CLEARANCE))
    print()

    failures = []

    print("SPAWN POSES")
    for rid, (x, y) in sorted(SPAWNS.items()):
        c = grid.clearance_at(x, y)
        ok = c >= SPAWN_CLEARANCE
        print("  %-6s (%6.2f,%6.2f)  clearance %.2f m   %s"
              % (rid, x, y, c, "ok" if ok else "TOO TIGHT"))
        if not ok:
            failures.append("spawn %s has only %.2f m" % (rid, c))
    print()

    print("STATIONS")
    for name, (x, y) in sorted(stations.items()):
        c = grid.clearance_at(x, y)
        ok = c >= PLANNING_CLEARANCE
        print("  %-16s (%6.2f,%6.2f)  clearance %.2f m   %s"
              % (name, x, y, c, "ok" if ok else "TOO TIGHT"))
        if not ok:
            failures.append("station %s has only %.2f m" % (name, c))
    print()

    print("ROUTABILITY  (every station pair, A* over the static map)")
    unroutable = []
    longest = (0.0, None)
    for a, b in itertools.combinations(sorted(stations), 2):
        length = planner.route_cost(stations[a], stations[b])
        if length is None:
            unroutable.append((a, b))
        elif length > longest[0]:
            longest = (length, (a, b))
    total = len(stations) * (len(stations) - 1) // 2
    print("  %d/%d station pairs routable" % (total - len(unroutable), total))
    if longest[1]:
        print("  longest route: %s -> %s, %.2f m"
              % (longest[1][0], longest[1][1], longest[0]))
    for a, b in unroutable:
        print("  UNROUTABLE: %s <-> %s" % (a, b))
        failures.append("no route %s <-> %s" % (a, b))
    print()

    print("TASK LIST")
    for spec in (cfg.get("tasks") or []):
        p, d = spec["pickup"], spec["dropoff"]
        if p not in stations or d not in stations:
            failures.append("task %s references an unknown station" % spec.get("id"))
            print("  %-5s UNKNOWN STATION (%s -> %s)" % (spec.get("id"), p, d))
            continue
        leg = planner.route_cost(stations[p], stations[d])
        print("  %-5s %-34s %6.2f m" % (spec.get("id"), spec.get("label", ""),
                                        leg if leg else float("nan")))
    print()

    if failures:
        print("FAILED (%d):" % len(failures))
        for f in failures:
            print("  - " + f)
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
