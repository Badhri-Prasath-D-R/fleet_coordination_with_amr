#!/usr/bin/env python3
"""Occupancy grid loaded straight from a map_server PGM/YAML pair.

Pure Python and pure stdlib -- no numpy, no scipy, no OpenCV. That is a
deliberate constraint: this module runs inside four robot processes on a
laptop under WSL2, and every extra dependency is one more thing that can
fail to be installed on demo day. Loading ``bcr_map.pgm`` (587 x 624) and
building a clearance field over it takes well under a second.

Two conventions worth stating, because they are the usual source of
silently-mirrored maps:

* map_server pixel ``(col, row)`` has row 0 at the **top**, while the YAML
  ``origin`` refers to the **bottom-left** corner. This module converts once
  on load and indexes everything bottom-up thereafter, so ``gy`` grows with
  world ``y`` and no flip survives past ``load_map()``.
* The stored grey level is not occupancy. For ``negate: 0``, occupancy is
  ``(255 - pixel) / 255``, which is why 205 (the conventional "unknown"
  value) scores 0.196 and is *free* under this map ``free_thresh: 0.25``.
"""

import math
import os

__all__ = ["OccupancyGrid", "load_map"]


# --------------------------------------------------------------- parsing

def _parse_map_yaml(path):
    """Read a map_server YAML. Uses PyYAML when present, else a small reader.

    The fallback exists so a missing python3-yaml cannot stop the fleet from
    launching; these files only ever contain scalars and one flow sequence.
    """
    try:
        import yaml
        with open(path, "r") as fh:
            return yaml.safe_load(fh)
    except ImportError:
        pass

    out = {}
    with open(path, "r") as fh:
        for line in fh:
            line = line.split("#", 1)[0].strip()
            if not line or ":" not in line:
                continue
            key, _, val = line.partition(":")
            val = val.strip()
            if val.startswith("["):
                out[key.strip()] = [float(v) for v in
                                    val.strip("[]").split(",") if v.strip()]
            else:
                try:
                    out[key.strip()] = float(val)
                except ValueError:
                    out[key.strip()] = val.strip().strip("\"").strip("\x27")
    return out


def _read_pgm(path):
    """Return ``(width, height, maxval, rows)`` for a binary or ASCII PGM."""
    with open(path, "rb") as fh:
        data = fh.read()

    pos = 0

    def token():
        nonlocal pos
        while True:
            while pos < len(data) and data[pos:pos + 1].isspace():
                pos += 1
            if pos < len(data) and data[pos:pos + 1] == b"#":
                while pos < len(data) and data[pos:pos + 1] not in (b"\n", b"\r"):
                    pos += 1
                continue
            break
        start = pos
        while pos < len(data) and not data[pos:pos + 1].isspace():
            pos += 1
        return data[start:pos]

    magic = token()
    width = int(token())
    height = int(token())
    maxval = int(token())

    if magic == b"P5":
        pos += 1                                   # exactly one whitespace byte
        wide = maxval > 255
        need = width * height * (2 if wide else 1)
        raw = data[pos:pos + need]
        if len(raw) < need:
            raise ValueError("%s: truncated PGM (%d/%d bytes)"
                             % (path, len(raw), need))
        if wide:                                   # 16-bit, big endian
            raw = bytes(raw[i] for i in range(0, need, 2))
        rows = [raw[r * width:(r + 1) * width] for r in range(height)]
    elif magic == b"P2":
        vals = [int(token()) for _ in range(width * height)]
        scale = 255.0 / maxval if maxval else 1.0
        flat = bytes(min(255, int(v * scale)) for v in vals)
        rows = [flat[r * width:(r + 1) * width] for r in range(height)]
    else:
        raise ValueError("%s: unsupported PGM magic %r" % (path, magic))

    return width, height, maxval, rows


# ------------------------------------------------------------------ grid

class OccupancyGrid:
    """A coarsened, bottom-up occupancy grid with a precomputed clearance field.

    ``blocked[i]`` is True when cell ``i`` contains any occupied source pixel.
    ``clearance[i]`` is the distance in metres from that cell to the nearest
    blocked cell, which is what lets the planner ask the only question that
    actually matters -- can a chassis this wide stand here -- with a single
    comparison instead of a footprint convolution per node expansion.
    """

    def __init__(self, width, height, resolution, origin, blocked):
        self.width = width
        self.height = height
        self.resolution = resolution
        self.origin_x, self.origin_y = origin[0], origin[1]
        self.blocked = blocked
        self.clearance = [0.0] * (width * height)
        self._build_clearance()

    # -------------------------------------------------------- conversions

    def index(self, gx, gy):
        return gy * self.width + gx

    def in_bounds(self, gx, gy):
        return 0 <= gx < self.width and 0 <= gy < self.height

    def world_to_cell(self, x, y):
        return (int(math.floor((x - self.origin_x) / self.resolution)),
                int(math.floor((y - self.origin_y) / self.resolution)))

    def cell_to_world(self, gx, gy):
        return (self.origin_x + (gx + 0.5) * self.resolution,
                self.origin_y + (gy + 0.5) * self.resolution)

    # ---------------------------------------------------------- queries

    def clearance_at(self, x, y):
        """Metres of free space around a world point; 0.0 outside the map."""
        gx, gy = self.world_to_cell(x, y)
        if not self.in_bounds(gx, gy):
            return 0.0
        return self.clearance[self.index(gx, gy)]

    def is_traversable(self, gx, gy, min_clearance):
        return (self.in_bounds(gx, gy)
                and self.clearance[self.index(gx, gy)] >= min_clearance)

    def nearest_traversable(self, x, y, min_clearance, search_radius=2.5):
        """Snap a world point onto the closest cell a robot can legally occupy.

        Task stations and goals get authored by hand and land a few
        centimetres inside a shelf leg often enough that failing outright
        would be the wrong behaviour; nudging to the nearest legal cell and
        reporting it is more useful than refusing the job.
        """
        gx, gy = self.world_to_cell(x, y)
        if self.is_traversable(gx, gy, min_clearance):
            return gx, gy
        rings = int(math.ceil(search_radius / self.resolution))
        for r in range(1, rings + 1):
            best, best_d = None, float("inf")
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    if max(abs(dx), abs(dy)) != r:
                        continue
                    cx, cy = gx + dx, gy + dy
                    if not self.is_traversable(cx, cy, min_clearance):
                        continue
                    d = dx * dx + dy * dy
                    if d < best_d:
                        best, best_d = (cx, cy), d
            if best is not None:
                return best
        return None

    def line_of_sight(self, gx0, gy0, gx1, gy1, min_clearance, forbidden=None):
        """Bresenham traversability check, used to shortcut A* zig-zags.

        ``forbidden`` is an optional set of cell indices that the ray may not
        cross even though the static map allows it -- the dynamic layer. It
        is not optional in practice: smoothing that ignores the peers will
        happily straighten a hard-won detour back through the robot it was
        planned to avoid.
        """
        dx = abs(gx1 - gx0)
        dy = abs(gy1 - gy0)
        sx = 1 if gx1 > gx0 else -1
        sy = 1 if gy1 > gy0 else -1
        err = dx - dy
        x, y = gx0, gy0
        while True:
            if not self.is_traversable(x, y, min_clearance):
                return False
            if forbidden and self.index(x, y) in forbidden:
                return False
            if x == gx1 and y == gy1:
                return True
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy

    # -------------------------------------------------------- clearance

    def _build_clearance(self):
        """Two-pass 5-7-11 chamfer distance transform (~2% error vs Euclidean).

        Cheaper and simpler than an exact EDT, and the error sits far below
        the coarse cell size, so it never changes a traversability verdict
        that a robot own safety margin would not already absorb.
        """
        w, h = self.width, self.height
        big = 10 ** 9
        d = [0 if b else big for b in self.blocked]

        fwd = ((-1, -2, 11), (-1, -1, 7), (-1, 0, 5), (-1, 1, 7), (-1, 2, 11),
               (-2, -1, 11), (-2, 1, 11), (0, -1, 5))
        bwd = tuple((-dy, -dx, c) for dy, dx, c in fwd)

        for gy in range(h):
            base = gy * w
            for gx in range(w):
                i = base + gx
                if d[i] == 0:
                    continue
                best = d[i]
                for dy, dx, cost in fwd:
                    ny, nx = gy + dy, gx + dx
                    if 0 <= ny < h and 0 <= nx < w:
                        v = d[ny * w + nx] + cost
                        if v < best:
                            best = v
                d[i] = best

        for gy in range(h - 1, -1, -1):
            base = gy * w
            for gx in range(w - 1, -1, -1):
                i = base + gx
                if d[i] == 0:
                    continue
                best = d[i]
                for dy, dx, cost in bwd:
                    ny, nx = gy + dy, gx + dx
                    if 0 <= ny < h and 0 <= nx < w:
                        v = d[ny * w + nx] + cost
                        if v < best:
                            best = v
                d[i] = best

        scale = self.resolution / 5.0
        self.clearance = [min(v, big) * scale for v in d]

    # ------------------------------------------------------------ misc

    def stats(self):
        return {
            "size": (self.width, self.height),
            "cells": self.width * self.height,
            "resolution": round(self.resolution, 3),
            "origin": (self.origin_x, self.origin_y),
            "blocked": sum(1 for b in self.blocked if b),
            "extent_x": (round(self.origin_x, 2),
                         round(self.origin_x + self.width * self.resolution, 2)),
            "extent_y": (round(self.origin_y, 2),
                         round(self.origin_y + self.height * self.resolution, 2)),
        }

    def __repr__(self):
        return ("<OccupancyGrid %dx%d @%.2fm origin=(%.2f,%.2f)>"
                % (self.width, self.height, self.resolution,
                   self.origin_x, self.origin_y))


def load_map(yaml_path, downsample=3, unknown_is_obstacle=False,
             border_is_obstacle=True):
    """Load a map_server YAML+PGM into a coarsened :class:`OccupancyGrid`.

    ``downsample`` trades planning resolution for A* speed. A block is
    blocked if *any* source pixel in it is occupied, so coarsening can only
    ever be conservative -- it never opens a gap that is not really there.
    """
    meta = _parse_map_yaml(yaml_path)
    image = meta.get("image", "map.pgm")
    if not os.path.isabs(image):
        image = os.path.join(os.path.dirname(os.path.abspath(yaml_path)), image)

    res = float(meta.get("resolution", 0.05))
    origin = meta.get("origin", [0.0, 0.0, 0.0])
    negate = int(meta.get("negate", 0))
    occ_thresh = float(meta.get("occupied_thresh", 0.65))
    free_thresh = float(meta.get("free_thresh", 0.25))

    width, height, _maxval, rows = _read_pgm(image)

    # Classify all 256 grey levels once, then use bytes.translate so the
    # per-pixel work happens in C rather than in a 366k-iteration Python loop.
    table = bytearray(256)
    for v in range(256):
        p = v / 255.0 if negate else (255 - v) / 255.0
        if p > occ_thresh:
            table[v] = 1
        elif p < free_thresh:
            table[v] = 0
        else:
            table[v] = 1 if unknown_is_obstacle else 0
    table = bytes(table)

    f = max(1, int(downsample))
    cw = (width + f - 1) // f
    ch = (height + f - 1) // f
    blocked = [False] * (cw * ch)

    for row in range(height):
        marks = rows[row].translate(table)
        if 1 not in marks:
            continue
        gy = (height - 1 - row) // f          # bottom-up coarse row
        base = gy * cw
        col = marks.find(1)
        while col != -1:
            blocked[base + col // f] = True
            col = marks.find(1, col + 1)

    if border_is_obstacle:
        for gx in range(cw):
            blocked[gx] = True
            blocked[(ch - 1) * cw + gx] = True
        for gy in range(ch):
            blocked[gy * cw] = True
            blocked[gy * cw + cw - 1] = True

    return OccupancyGrid(cw, ch, res * f, (origin[0], origin[1]), blocked)
