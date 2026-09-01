#!/usr/bin/env python3
"""Score a proximity log: the old distance rule against the predictive rule.

    python3 replay_proximity_log.py p2p_conflict_log.txt

Run it on a log captured before the new layer was in, then on one captured
after, and the difference is a number rather than an adjective.

It reads both generations of log. The original format recorded only
positions, so velocities have to be recovered by finite-differencing the
samples -- which is exactly the handicap that made the original encounters
impossible to diagnose after the fact. The new format records ``vel=(vx,vy)``
and ``yaw=`` on every line, and when those are present they are used
directly; the header of the report says which happened.

Parsing is field by field rather than one monolithic pattern. That is not
fussiness: the original regex expected ``prio=N | `` immediately, while the
logger actually emitted ``prio=N next_wp=(...) | ``, so it matched no lines
at all and the tool reported an empty file.
"""

import math
import re
import sys
from datetime import datetime

# ---------------------------------------------------------------- tuning
#
# Chassis from bcr_bot.xacro: 0.90 x 0.64 m, modelled as a capsule of
# radius 0.32 with a 0.13 m spine -- the exact rounded hull of that
# rectangle. The margin is metres of air between hulls, which is the
# quantity a safety requirement is actually written in.

FOOT_LENGTH = 0.90
FOOT_WIDTH = 0.64
SAFETY_MARGIN = 0.25
HORIZON = 8.0
OLD_TRIGGER_M = 1.5          # the distance rule this replaces
GAP_S = 2.0                  # split encounters on log gaps larger than this
VEL_WIN = 5                  # +/- samples for the finite-difference fallback

RADIUS = 0.5 * FOOT_WIDTH
HALF_LEN = max(0.0, 0.5 * (FOOT_LENGTH - FOOT_WIDTH))

TIME_RE = re.compile(r"^\[(\d{2}:\d{2}:\d{2}\.\d+)\]")
TAG_RE = re.compile(r"\[PROXIMITY\]")
PAIR_RE = re.compile(r"(\w+)\s*<->\s*(\w+)")
DIST_RE = re.compile(r"dist=(-?[\d.]+)m")
FIELD_RE = re.compile(r"(\w+):\s*pos=\((-?[\d.]+),\s*(-?[\d.]+)\)")
YAW_RE = re.compile(r"yaw=(-?[\d.]+)")
VEL_RE = re.compile(r"vel=\((-?[\d.]+),\s*(-?[\d.]+)\)")


# ------------------------------------------------------------- geometry

def cpa(pa, va, pb, vb):
    rx, ry = pb[0] - pa[0], pb[1] - pa[1]
    vx, vy = vb[0] - va[0], vb[1] - va[1]
    vv = vx * vx + vy * vy
    if vv < 1e-6:
        return math.inf, math.hypot(rx, ry)
    t = -(rx * vx + ry * vy) / vv
    if t < 0.0:
        return -1.0, math.hypot(rx, ry)
    return t, math.hypot(rx + vx * t, ry + vy * t)


def _clamp01(v):
    return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)


def seg_seg_distance(p1, q1, p2, q2):
    d1x, d1y = q1[0] - p1[0], q1[1] - p1[1]
    d2x, d2y = q2[0] - p2[0], q2[1] - p2[1]
    rx, ry = p1[0] - p2[0], p1[1] - p2[1]
    a = d1x * d1x + d1y * d1y
    e = d2x * d2x + d2y * d2y
    f = d2x * rx + d2y * ry
    eps = 1e-9
    if a <= eps and e <= eps:
        return math.hypot(rx, ry)
    if a <= eps:
        s, t = 0.0, _clamp01(f / e)
    else:
        c = d1x * rx + d1y * ry
        if e <= eps:
            t, s = 0.0, _clamp01(-c / a)
        else:
            b = d1x * d2x + d1y * d2y
            denom = a * e - b * b
            s = _clamp01((b * f - c * e) / denom) if denom > eps else 0.0
            t = (b * s + f) / e
            if t < 0.0:
                t, s = 0.0, _clamp01(-c / a)
            elif t > 1.0:
                t, s = 1.0, _clamp01((b - c) / a)
    cx = (p1[0] + d1x * s) - (p2[0] + d2x * t)
    cy = (p1[1] + d1y * s) - (p2[1] + d2y * t)
    return math.hypot(cx, cy)


def hull_gap(pose_a, pose_b):
    ax, ay, ayaw = pose_a
    bx, by, byaw = pose_b
    adx, ady = HALF_LEN * math.cos(ayaw), HALF_LEN * math.sin(ayaw)
    bdx, bdy = HALF_LEN * math.cos(byaw), HALF_LEN * math.sin(byaw)
    return seg_seg_distance((ax - adx, ay - ady), (ax + adx, ay + ady),
                            (bx - bdx, by - bdy), (bx + bdx, by + bdy)) \
        - 2.0 * RADIUS


def sweep(pose_a, va, pose_b, vb, horizon=HORIZON, step=0.25,
          threshold=SAFETY_MARGIN):
    t_hit, best, t_best = None, float("inf"), 0.0
    n = int(round(horizon / step))
    for i in range(n + 1):
        t = i * step
        pa = (pose_a[0] + va[0] * t, pose_a[1] + va[1] * t, pose_a[2])
        pb = (pose_b[0] + vb[0] * t, pose_b[1] + vb[1] * t, pose_b[2])
        gap = hull_gap(pa, pb)
        if gap < best:
            best, t_best = gap, t
        if t_hit is None and gap < threshold:
            t_hit = t
    return t_hit, best, t_best


# --------------------------------------------------------------- parsing

def parse(path):
    rows = []
    logged_velocity = 0
    with open(path, "r", errors="replace") as fh:
        for raw in fh:
            line = raw.strip()
            if not TAG_RE.search(line):
                continue
            pair = PAIR_RE.search(line)
            stamp = TIME_RE.match(line)
            dist = DIST_RE.search(line)
            if not (pair and stamp and dist):
                continue

            # Split at the pipes and read the two robot blocks by name, so
            # extra fields between them cannot break the parse.
            blocks = {}
            for chunk in line.split("|"):
                m = FIELD_RE.search(chunk)
                if not m:
                    continue
                yaw = YAW_RE.search(chunk)
                vel = VEL_RE.search(chunk)
                blocks[m.group(1)] = {
                    "pos": (float(m.group(2)), float(m.group(3))),
                    "yaw": float(yaw.group(1)) if yaw else None,
                    "vel": ((float(vel.group(1)), float(vel.group(2)))
                            if vel else None),
                }

            a_id, b_id = pair.group(1), pair.group(2)
            if a_id not in blocks or b_id not in blocks:
                continue
            if blocks[a_id]["vel"] is not None:
                logged_velocity += 1

            rows.append({
                "t": datetime.strptime(stamp.group(1), "%H:%M:%S.%f"),
                "d": float(dist.group(1)),
                "pair": (a_id, b_id),
                "a": blocks[a_id],
                "b": blocks[b_id],
            })

    if not rows:
        sys.exit("no [PROXIMITY] lines found in %s" % path)

    t0 = rows[0]["t"]
    for r in rows:
        r["ts"] = (r["t"] - t0).total_seconds()
    return rows, logged_velocity


def split_encounters(rows):
    out = [[rows[0]]]
    for prev, cur in zip(rows, rows[1:]):
        if cur["ts"] - prev["ts"] > GAP_S or cur["pair"] != prev["pair"]:
            out.append([])
        out[-1].append(cur)
    return out


def finite_difference(seq, i, key):
    lo, hi = max(0, i - VEL_WIN), min(len(seq) - 1, i + VEL_WIN)
    dt = seq[hi]["ts"] - seq[lo]["ts"]
    if dt <= 0:
        return (0.0, 0.0)
    return ((seq[hi][key]["pos"][0] - seq[lo][key]["pos"][0]) / dt,
            (seq[hi][key]["pos"][1] - seq[lo][key]["pos"][1]) / dt)


def velocity_of(seq, i, key):
    v = seq[i][key]["vel"]
    return v if v is not None else finite_difference(seq, i, key)


def heading_of(seq, i, key, vel):
    yaw = seq[i][key]["yaw"]
    if yaw is not None:
        return yaw
    if math.hypot(*vel) > 1e-3:
        return math.atan2(vel[1], vel[0])
    return 0.0


# ----------------------------------------------------------------- main

def main(path):
    rows, logged_velocity = parse(path)
    encounters = split_encounters(rows)
    source = ("logged velocities" if logged_velocity > len(rows) // 2
              else "velocities recovered by finite difference "
                   "(old-format log: positions only)")

    print("file             : %s" % path)
    print("samples          : %d in %d encounters" % (len(rows), len(encounters)))
    print("velocity source  : %s" % source)
    print("chassis          : %.2f x %.2f m as a capsule (r=%.2f, spine=%.2f)"
          % (FOOT_LENGTH, FOOT_WIDTH, RADIUS, HALF_LEN))
    print("required air gap : %.2f m between hulls    horizon: %.0f s"
          % (SAFETY_MARGIN, HORIZON))
    print("old rule         : brake whenever centre distance < %.2f m"
          % OLD_TRIGGER_M)
    print()

    total_old = total_new = 0
    wasted = 0
    real = 0
    lead_times = []

    for k, enc in enumerate(encounters, 1):
        old = sum(1 for r in enc if r["d"] < OLD_TRIGGER_M)
        new = 0
        first = None
        min_pred_gap = float("inf")
        min_actual_gap = float("inf")

        for i, r in enumerate(enc):
            va = velocity_of(enc, i, "a")
            vb = velocity_of(enc, i, "b")
            ya = heading_of(enc, i, "a", va)
            yb = heading_of(enc, i, "b", vb)
            pa = (r["a"]["pos"][0], r["a"]["pos"][1], ya)
            pb = (r["b"]["pos"][0], r["b"]["pos"][1], yb)

            min_actual_gap = min(min_actual_gap, hull_gap(pa, pb))
            t_cpa, _d_cpa = cpa(r["a"]["pos"], va, r["b"]["pos"], vb)
            t_hit, gap_min, _ = sweep(pa, va, pb, vb)
            min_pred_gap = min(min_pred_gap, gap_min)

            if t_hit is not None:
                new += 1
                if first is None:
                    first = (r["ts"] - enc[0]["ts"], t_hit, gap_min, t_cpa)

        total_old += old
        total_new += new
        if new:
            real += 1
            if first:
                lead_times.append(first[1])
        else:
            # Every brake the old rule applied here was applied to an
            # encounter that could not have ended in contact.
            wasted += old

        verdict = "REAL CONFLICT" if new else "false positive"
        print("encounter %d  %s -> %s  (%.1fs, %s)   [%s]"
              % (k, enc[0]["t"].strftime("%H:%M:%S"),
                 enc[-1]["t"].strftime("%H:%M:%S"),
                 enc[-1]["ts"] - enc[0]["ts"],
                 " <-> ".join(enc[0]["pair"]), verdict))
        print("   closest centre separation   : %.2f m"
              % min(r["d"] for r in enc))
        print("   closest hull gap, actual    : %.2f m" % min_actual_gap)
        print("   closest hull gap, predicted : %.2f m" % min_pred_gap)
        print("   old rule                    : %d brake samples of %d"
              % (old, len(enc)))
        print("   predictive rule             : %d brake samples of %d"
              % (new, len(enc)))
        if first:
            off, t_hit, gap, t_cpa = first
            print("   earliest warning            : %.1fs into the encounter, "
                  "%.1fs before contact, gap would be %.2f m (t_cpa %.1fs)"
                  % (off, t_hit, gap, t_cpa))
        print()

    print("-" * 72)
    print("TOTAL")
    print("   brake samples, old rule            : %d" % total_old)
    print("   brake samples, predictive rule     : %d" % total_new)
    if total_old:
        print("   old-rule braking spent on encounters")
        print("   that could not have ended in contact: %d  (%.0f%% of it)"
              % (wasted, 100.0 * wasted / total_old))
    print("   genuine conflicts                  : %d of %d encounters"
          % (real, len(encounters)))
    if lead_times:
        print("   mean warning lead time             : %.1f s before contact"
              % (sum(lead_times) / len(lead_times)))
    print()
    print("The headline number to quote is the third line: braking the old")
    print("rule applied to passes that were never going to touch. It is a")
    print("property of the encounters in this file, so run it on your own")
    print("captured log rather than reusing a figure from anywhere else.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "p2p_conflict_log.txt")
