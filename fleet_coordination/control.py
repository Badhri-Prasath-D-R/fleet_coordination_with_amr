#!/usr/bin/env python3
"""Path following and the speed governor, kept out of the ROS nodes.

Both live here rather than inside their nodes for one reason: the offline
simulator in ``scripts/simulate_fleet.py`` runs the *same* code the robots
run. A simulator that reimplements the controller it is meant to validate
proves nothing except that two pieces of code were written by the same
person on the same day.

Nothing in this module imports rclpy, so it is also directly unit-testable.
"""

import math

from fleet_coordination.geometry import clamp, wrap_angle

__all__ = ["FollowResult", "PathFollower", "speed_governor"]


def speed_governor(t_hit, brake_ttc, stop_ttc, min_scale):
    """Continuous speed scale from time-to-conflict.

    Full speed beyond ``brake_ttc``, zero inside ``stop_ttc``, linear
    between. The snap to zero below ``min_scale`` is the important part:
    crawling accomplishes nothing except holding the trigger threshold,
    which is exactly the failure mode that froze two robots 1.12 m apart
    for 4.5 seconds while both of them kept moving.
    """
    if t_hit is None:
        return 1.0
    s = clamp((t_hit - stop_ttc) / max(brake_ttc - stop_ttc, 1e-3), 0.0, 1.0)
    return 0.0 if s < min_scale else s


class FollowResult:
    """What the follower wants to do this tick."""

    __slots__ = ("v", "w", "arrived", "remaining", "alpha", "spinning",
                 "lookahead")

    def __init__(self, v, w, arrived, remaining, alpha, spinning, lookahead):
        self.v = v
        self.w = w
        self.arrived = arrived
        self.remaining = remaining
        self.alpha = alpha
        self.spinning = spinning
        self.lookahead = lookahead


class PathFollower:
    """Pure pursuit with an acceleration ramp and a curvature speed limit.

    The index into the path only ever moves forward. Without that a route
    that doubles back near itself -- which is precisely what a reroute
    around a blocking peer produces -- can snap the follower onto the
    earlier leg and send the robot back the way it came.
    """

    def __init__(self, max_linear=0.40, max_angular=1.00, max_accel=0.35,
                 lookahead_m=0.80, lookahead_gain=0.9, lookahead_min=0.45,
                 lookahead_max=1.60, spin_threshold_rad=math.radians(35.0),
                 spin_gain=1.6, goal_tolerance=0.25, final_approach_m=1.0):
        self.max_v = max_linear
        self.max_w = max_angular
        self.max_a = max_accel
        self.look_base = lookahead_m
        self.look_gain = lookahead_gain
        self.look_min = lookahead_min
        self.look_max = lookahead_max
        self.spin_threshold = spin_threshold_rad
        self.spin_gain = spin_gain
        self.goal_tol = goal_tolerance
        self.final_approach = final_approach_m
        self.index = 0

    def reset(self):
        self.index = 0

    # ------------------------------------------------------------ geometry

    def project(self, pose, path):
        """Closest point on the path at or after the current index.

        Returns ``(segment, point, remaining_arclength)``.
        """
        self.index = min(max(0, self.index), len(path) - 2)
        px, py = pose[0], pose[1]
        best_seg, best_pt, best_d = self.index, path[self.index], float("inf")

        for i in range(self.index, len(path) - 1):
            ax, ay = path[i]
            bx, by = path[i + 1]
            dx, dy = bx - ax, by - ay
            seg2 = dx * dx + dy * dy
            t = 0.0 if seg2 < 1e-12 else clamp(
                ((px - ax) * dx + (py - ay) * dy) / seg2, 0.0, 1.0)
            cx, cy = ax + dx * t, ay + dy * t
            d = math.hypot(px - cx, py - cy)
            if d < best_d:
                best_seg, best_pt, best_d = i, (cx, cy), d

        self.index = best_seg
        remaining = math.hypot(path[best_seg + 1][0] - best_pt[0],
                               path[best_seg + 1][1] - best_pt[1])
        for i in range(best_seg + 1, len(path) - 1):
            remaining += math.hypot(path[i + 1][0] - path[i][0],
                                    path[i + 1][1] - path[i][1])
        return best_seg, best_pt, remaining

    @staticmethod
    def lookahead_point(path, seg, point, distance):
        """Walk ``distance`` metres along the path from ``point``."""
        left = distance
        cur = point
        for i in range(seg, len(path) - 1):
            nxt = path[i + 1]
            step = math.hypot(nxt[0] - cur[0], nxt[1] - cur[1])
            if step >= left:
                if step < 1e-9:
                    return nxt
                f = left / step
                return (cur[0] + (nxt[0] - cur[0]) * f,
                        cur[1] + (nxt[1] - cur[1]) * f)
            left -= step
            cur = nxt
        return path[-1]

    # ---------------------------------------------------------------- step

    def step(self, pose, measured_speed, path, goal, prev_cmd_v, dt):
        """One control tick. Returns a :class:`FollowResult`."""
        if not path or len(path) < 2:
            return FollowResult(0.0, 0.0, False, 0.0, 0.0, False, None)

        seg, point, remaining = self.project(pose, path)
        goal_dist = math.hypot(goal[0] - pose[0], goal[1] - pose[1])
        if goal_dist <= self.goal_tol or remaining <= self.goal_tol:
            return FollowResult(0.0, 0.0, True, remaining, 0.0, False, None)

        look = clamp(self.look_base + self.look_gain * measured_speed,
                     self.look_min, self.look_max)
        target = self.lookahead_point(path, seg, point, look)
        alpha = wrap_angle(math.atan2(target[1] - pose[1],
                                      target[0] - pose[0]) - pose[2])

        if abs(alpha) > self.spin_threshold:
            # Too far off course to drive: turn first. Pure rotation is also
            # the one motion the collision layer will still permit while it
            # is holding this robot stationary.
            w = clamp(self.spin_gain * alpha, -self.max_w, self.max_w)
            return FollowResult(0.0, w, False, remaining, alpha, True, target)

        v_target = self.max_v
        stop_dist = min(remaining, goal_dist)
        if stop_dist < self.final_approach:
            v_target = min(v_target,
                           max(0.08, self.max_v * stop_dist / self.final_approach))

        curvature = 2.0 * math.sin(alpha) / max(look, 1e-3)
        if abs(curvature) > 1e-3:
            v_target = min(v_target, self.max_w / abs(curvature))

        # Ramp from whichever is lower, our last command or what the base
        # actually achieved. If the collision layer zeroed us, this restarts
        # the ramp from a standstill instead of snapping back to full speed.
        base = min(prev_cmd_v, measured_speed + 0.10)
        v = clamp(v_target, base - self.max_a * dt, base + self.max_a * dt)
        v = clamp(v, 0.0, self.max_v)
        w = clamp(curvature * v, -self.max_w, self.max_w)
        return FollowResult(v, w, False, remaining, alpha, False, target)
