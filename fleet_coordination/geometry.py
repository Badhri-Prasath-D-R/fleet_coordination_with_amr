#!/usr/bin/env python3
"""Geometry and prediction primitives shared by every fleet node.

Pure Python, no ROS and no numpy, so it can be unit-tested and replayed
offline against a log without sourcing a workspace.

The two ideas that matter here:

* Closest point of approach (CPA) turns two positions and two velocities
  into "how close will these get, and when". Raw separation cannot tell
  "converging head-on at 0.5 m/s" from "already driving apart" -- both look
  like dist=1.20m. Relative velocity can.

* A robot is not a point and usually not a disc either. The bcr_bot chassis
  is 0.90 x 0.64 m; a single circumscribed disc (r=0.55) brakes for passes
  that were never going to touch, and a single inscribed disc (r=0.32) will
  happily drive its own corners into a peer. A capsule -- a segment of
  half-length (L-W)/2 swept by a circle of radius W/2 -- is the exact
  rounded hull of that rectangle and costs one segment-to-segment distance.
"""

import math

__all__ = [
    "Footprint",
    "angle_diff",
    "capsule_clearance",
    "clamp",
    "closest_point_of_approach",
    "predict_pose",
    "segment_segment_distance",
    "sweep_clearance",
    "wrap_angle",
    "yaw_from_quat",
]

_EPS = 1e-9


# ---------------------------------------------------------------- scalars

def clamp(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)


def wrap_angle(a):
    """Fold an angle into (-pi, pi]."""
    return math.atan2(math.sin(a), math.cos(a))


def angle_diff(a, b):
    """Unsigned smallest angle between two headings, in [0, pi]."""
    return abs(wrap_angle(a - b))


def yaw_from_quat(q):
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


# ------------------------------------------------------------------- CPA

def closest_point_of_approach(pa, va, pb, vb):
    """Return ``(t_cpa, d_cpa)`` for two point masses in straight-line motion.

    ``t_cpa`` is seconds until minimum separation:
        -1.0      the closest approach is already behind us (separating)
        math.inf  no relative motion, separation is constant
    ``d_cpa`` is the separation reached at ``t_cpa``, in metres.

    This is the cheap first filter. It is exact for point masses and is what
    decides "is this pair converging at all", which alone removes the entire
    class of false positives where two robots are driving apart.
    """
    rx, ry = pb[0] - pa[0], pb[1] - pa[1]
    vx, vy = vb[0] - va[0], vb[1] - va[1]
    vv = vx * vx + vy * vy

    if vv < 1e-6:
        return math.inf, math.hypot(rx, ry)

    t = -(rx * vx + ry * vy) / vv
    if t < 0.0:
        return -1.0, math.hypot(rx, ry)

    return t, math.hypot(rx + vx * t, ry + vy * t)


def predict_pose(pose, vx, vy, omega, t):
    """Constant-velocity, constant-turn-rate extrapolation of a pose."""
    return (pose[0] + vx * t, pose[1] + vy * t, pose[2] + omega * t)


# ------------------------------------------------------------- footprints

def _clamp01(v):
    return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)


def segment_segment_distance(p1, q1, p2, q2):
    """Shortest distance between segments p1q1 and p2q2 (Ericson, RTCD 5.1.9)."""
    d1x, d1y = q1[0] - p1[0], q1[1] - p1[1]
    d2x, d2y = q2[0] - p2[0], q2[1] - p2[1]
    rx, ry = p1[0] - p2[0], p1[1] - p2[1]

    a = d1x * d1x + d1y * d1y          # |d1|^2
    e = d2x * d2x + d2y * d2y          # |d2|^2
    f = d2x * rx + d2y * ry

    if a <= _EPS and e <= _EPS:                       # both degenerate
        return math.hypot(rx, ry)

    if a <= _EPS:                                     # first is a point
        s, t = 0.0, _clamp01(f / e)
    else:
        c = d1x * rx + d1y * ry
        if e <= _EPS:                                 # second is a point
            t, s = 0.0, _clamp01(-c / a)
        else:
            b = d1x * d2x + d1y * d2y
            denom = a * e - b * b
            s = _clamp01((b * f - c * e) / denom) if denom > _EPS else 0.0
            t = (b * s + f) / e
            if t < 0.0:
                t, s = 0.0, _clamp01(-c / a)
            elif t > 1.0:
                t, s = 1.0, _clamp01((b - c) / a)

    cx = (p1[0] + d1x * s) - (p2[0] + d2x * t)
    cy = (p1[1] + d1y * s) - (p2[1] + d2y * t)
    return math.hypot(cx, cy)


class Footprint:
    """Capsule cover of a rectangular chassis.

    ``radius`` is half the chassis width and ``half_len`` is (L - W) / 2, so
    sweeping a disc of ``radius`` along the axis segment reproduces a
    rounded rectangle exactly ``length`` long and ``width`` wide. For a
    square-ish or round robot ``half_len`` collapses to 0 and the capsule
    degenerates to the familiar disc -- no special case needed.
    """

    __slots__ = ("length", "width", "radius", "half_len")

    def __init__(self, length, width):
        self.length = float(length)
        self.width = float(width)
        self.radius = 0.5 * self.width
        self.half_len = max(0.0, 0.5 * (self.length - self.width))

    def axis(self, x, y, yaw):
        """Endpoints of the capsule's spine for a pose."""
        dx = self.half_len * math.cos(yaw)
        dy = self.half_len * math.sin(yaw)
        return (x - dx, y - dy), (x + dx, y + dy)

    def __repr__(self):
        return (f"Footprint(L={self.length:.2f}, W={self.width:.2f}, "
                f"r={self.radius:.2f}, half_len={self.half_len:.2f})")


def capsule_clearance(fp_a, pose_a, fp_b, pose_b):
    """Gap between two chassis hulls, in metres. Negative means overlap.

    This is the quantity a safety margin should actually be expressed in --
    "keep 25 cm of air between the robots" -- rather than a centre-to-centre
    distance that silently means different things depending on whether the
    pair is passing nose-to-nose or shoulder-to-shoulder.
    """
    a0, a1 = fp_a.axis(pose_a[0], pose_a[1], pose_a[2])
    b0, b1 = fp_b.axis(pose_b[0], pose_b[1], pose_b[2])
    return segment_segment_distance(a0, a1, b0, b1) - fp_a.radius - fp_b.radius


def sweep_clearance(fp_a, pose_a, twist_a, fp_b, pose_b, twist_b,
                    horizon, step, threshold):
    """Roll both robots forward and report the worst clearance on the way.

    ``twist`` is ``(vx, vy, omega)`` in the world frame. Returns

        (t_hit, min_clearance, t_at_min)

    ``t_hit`` is the first sampled time at which the hulls come within
    ``threshold`` of each other, or ``None`` if they never do inside the
    horizon. Sampling the whole horizon rather than testing only the
    point-mass CPA instant matters for turning robots and for long chassis,
    where the moment of least *hull* clearance is not the moment of least
    *centre* separation.
    """
    ax, ay, ayaw = pose_a
    bx, by, byaw = pose_b
    avx, avy, aw = twist_a
    bvx, bvy, bw = twist_b

    t_hit = None
    best = float("inf")
    t_best = 0.0

    n = max(1, int(round(horizon / step)))
    for i in range(n + 1):
        t = i * step
        pa = (ax + avx * t, ay + avy * t, ayaw + aw * t)
        pb = (bx + bvx * t, by + bvy * t, byaw + bw * t)
        c = capsule_clearance(fp_a, pa, fp_b, pb)
        if c < best:
            best, t_best = c, t
        if t_hit is None and c < threshold:
            t_hit = t
            # keep sweeping: we still want the true minimum for reporting
    return t_hit, best, t_best
