#!/usr/bin/env python3
"""A* over the static map, plus a dynamic layer holding the peers.

Rerouting is the layer that makes the whole escalation ladder terminate. A
speed governor can only ever answer "slow down"; slowing down does not
resolve a head-on encounter in a 1.8 m aisle, it just decides who arrives at
the stalemate first. The way out is a different route, which means the
planner has to know two things the static map cannot tell it: where the
other robots are right now, and where they intend to be.

Peers enter the search as a *cost* layer rather than as walls:

* ``hard`` obstacles (a robot parked across the only aisle) are impassable,
  which forces the search to find a genuinely different corridor or to
  report failure honestly so the task layer can hand the job to someone else.
* ``soft`` obstacles merely make cells expensive, which nudges routes away
  from congestion before it becomes a conflict. This is the cheapest kind of
  collision avoidance there is: the one that happens minutes early, on a
  route that never brings the two robots into the same aisle at all.

Clearance also enters the cost, so paths run down the middle of an aisle
rather than shaving the shelf legs. That single term removes most of the
marginal encounters that a purely shortest-path route would create.
"""

import heapq
import math

__all__ = ["DynamicObstacle", "Planner"]

_SQRT2 = math.sqrt(2.0)

# 8-connected neighbourhood as (dx, dy, step_in_cells)
_NEIGHBOURS = (
    (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
    (1, 1, _SQRT2), (1, -1, _SQRT2), (-1, 1, _SQRT2), (-1, -1, _SQRT2),
)


class DynamicObstacle:
    """A peer, projected onto the planning grid.

    ``radius`` is the influence radius, not the chassis radius: for a soft
    obstacle it is how far out the route should start bending away, and for
    a hard one it is the region the route may not enter at all.
    """

    __slots__ = ("x", "y", "radius", "hard", "weight", "rid")

    def __init__(self, x, y, radius, hard=False, weight=4.0, rid=None):
        self.x = float(x)
        self.y = float(y)
        self.radius = float(radius)
        self.hard = bool(hard)
        self.weight = float(weight)
        self.rid = rid

    def __repr__(self):
        return ("<DynObs %s (%.2f,%.2f) r=%.2f %s>"
                % (self.rid or "?", self.x, self.y, self.radius,
                   "HARD" if self.hard else "soft"))


class Planner:
    """Grid A* with a clearance preference and an injectable dynamic layer."""

    def __init__(self, grid, min_clearance=0.42, prefer_clearance=0.90,
                 clearance_weight=0.6, max_expansions=120000,
                 shortcut_cost_limit=1.0):
        self.grid = grid
        self.min_clearance = float(min_clearance)
        self.prefer_clearance = float(prefer_clearance)
        self.clearance_weight = float(clearance_weight)
        self.max_expansions = int(max_expansions)
        # Soft cells at least this expensive are treated as no-go by the
        # smoother, so a route bent around congestion stays bent.
        self.shortcut_cost_limit = float(shortcut_cost_limit)
        self._cost_cache = {}

    # ------------------------------------------------------ dynamic layer

    def _rasterise(self, dynamic):
        """Turn world-space obstacles into per-cell (extra_cost, blocked) maps.

        Done once per plan rather than per node expansion; with a handful of
        peers this touches a few hundred cells and keeps the inner loop of
        the search down to a dictionary lookup.
        """
        grid = self.grid
        extra = {}
        blocked = set()
        for obs in dynamic:
            cells = int(math.ceil(obs.radius / grid.resolution))
            cgx, cgy = grid.world_to_cell(obs.x, obs.y)
            for dy in range(-cells, cells + 1):
                for dx in range(-cells, cells + 1):
                    gx, gy = cgx + dx, cgy + dy
                    if not grid.in_bounds(gx, gy):
                        continue
                    wx, wy = grid.cell_to_world(gx, gy)
                    d = math.hypot(wx - obs.x, wy - obs.y)
                    if d > obs.radius:
                        continue
                    idx = grid.index(gx, gy)
                    if obs.hard:
                        blocked.add(idx)
                    else:
                        falloff = 1.0 - d / obs.radius
                        cost = obs.weight * falloff * falloff
                        if cost > extra.get(idx, 0.0):
                            extra[idx] = cost
        return extra, blocked

    # ------------------------------------------------------------- search

    def plan(self, start_xy, goal_xy, dynamic=(), relax_goal=True):
        """Return a list of world ``(x, y)`` waypoints, or ``None``.

        The returned path always starts at the true start position and ends
        at the true goal position, with the grid-snapped route in between,
        so a follower never has to reason about the discretisation.
        """
        grid = self.grid
        start = grid.nearest_traversable(start_xy[0], start_xy[1],
                                         self.min_clearance)
        if start is None:
            return None

        goal = grid.nearest_traversable(goal_xy[0], goal_xy[1],
                                        self.min_clearance)
        if goal is None:
            return None

        extra, hard_blocked = self._rasterise(dynamic)

        # A robot standing inside its own hard-blocked halo must still be
        # able to plan its way out, so the start cell is always passable.
        hard_blocked.discard(grid.index(*start))

        if grid.index(*goal) in hard_blocked:
            if not relax_goal:
                return None
            hard_blocked.discard(grid.index(*goal))

        res = grid.resolution
        w = grid.width
        gx1, gy1 = goal
        start_idx = grid.index(*start)
        goal_idx = grid.index(*goal)

        def heuristic(gx, gy):
            dx, dy = abs(gx - gx1), abs(gy - gy1)
            lo, hi = (dx, dy) if dx < dy else (dy, dx)
            return (hi - lo + _SQRT2 * lo) * res

        open_heap = [(heuristic(*start), 0.0, start_idx)]
        came = {}
        best = {start_idx: 0.0}
        closed = set()
        expansions = 0

        while open_heap:
            _f, g, idx = heapq.heappop(open_heap)
            if idx in closed:
                continue
            if idx == goal_idx:
                break
            closed.add(idx)
            expansions += 1
            if expansions > self.max_expansions:
                return None

            cgy, cgx = divmod(idx, w)
            for dx, dy, step in _NEIGHBOURS:
                ngx, ngy = cgx + dx, cgy + dy
                if not grid.is_traversable(ngx, ngy, self.min_clearance):
                    continue
                nidx = grid.index(ngx, ngy)
                if nidx in closed or nidx in hard_blocked:
                    continue

                # Diagonals may not cut a corner between two blocked cells.
                if dx and dy:
                    if not (grid.is_traversable(cgx + dx, cgy,
                                                self.min_clearance)
                            and grid.is_traversable(cgx, cgy + dy,
                                                    self.min_clearance)):
                        continue

                cost = step * res
                clear = grid.clearance[nidx]
                if clear < self.prefer_clearance:
                    deficit = (self.prefer_clearance - clear) / self.prefer_clearance
                    cost += self.clearance_weight * deficit * step * res
                cost += extra.get(nidx, 0.0) * step * res

                ng = g + cost
                if ng < best.get(nidx, float("inf")):
                    best[nidx] = ng
                    came[nidx] = idx
                    heapq.heappush(open_heap,
                                   (ng + heuristic(ngx, ngy), ng, nidx))
        else:
            return None                       # open list exhausted, no route

        # ------------------------------------------------------ reconstruct
        cells = []
        idx = goal_idx
        while idx != start_idx:
            cells.append(divmod(idx, w)[::-1])       # (gx, gy)
            idx = came[idx]
        cells.append(start)
        cells.reverse()

        # Smoothing has to respect the same dynamic layer the search did.
        avoid = set(hard_blocked)
        avoid.update(i for i, c in extra.items() if c >= self.shortcut_cost_limit)
        cells = self.shortcut(cells, forbidden=avoid)
        path = [grid.cell_to_world(gx, gy) for gx, gy in cells]
        path[0] = (start_xy[0], start_xy[1])
        path[-1] = (goal_xy[0], goal_xy[1])
        if len(path) < 2:                     # start and goal share a cell
            path = [(start_xy[0], start_xy[1]), (goal_xy[0], goal_xy[1])]
        return self._dedupe(path)

    # ---------------------------------------------------------- polishing

    def shortcut(self, cells, forbidden=None):
        """Collapse the grid staircase into long straight runs.

        Greedy line-of-sight smoothing. Without it, an 8-connected path
        through an aisle alternates between axis and diagonal moves, and a
        pure-pursuit follower turns that into visible weaving.
        """
        if len(cells) <= 2:
            return list(cells)
        out = [cells[0]]
        i = 0
        n = len(cells)
        while i < n - 1:
            j = n - 1
            while j > i + 1:
                if self.grid.line_of_sight(cells[i][0], cells[i][1],
                                           cells[j][0], cells[j][1],
                                           self.min_clearance, forbidden):
                    break
                j -= 1
            out.append(cells[j])
            i = j
        return out

    @staticmethod
    def _dedupe(path, eps=1e-3):
        out = [path[0]]
        for p in path[1:]:
            if math.hypot(p[0] - out[-1][0], p[1] - out[-1][1]) > eps:
                out.append(p)
        return out if len(out) > 1 else path

    # ------------------------------------------------------------ queries

    @staticmethod
    def path_length(path):
        if not path or len(path) < 2:
            return 0.0
        return sum(math.hypot(b[0] - a[0], b[1] - a[1])
                   for a, b in zip(path, path[1:]))

    def route_cost(self, start_xy, goal_xy, cache=True):
        """Length of the free-space route between two points, for bidding.

        Cached on grid cells: an auction re-prices the same station pairs
        over and over, and a bid that takes 200 ms to compute is a bid that
        arrives after the window has closed.
        """
        key = None
        if cache:
            key = (self.grid.world_to_cell(*start_xy),
                   self.grid.world_to_cell(*goal_xy))
            hit = self._cost_cache.get(key)
            if hit is not None:
                return hit

        path = self.plan(start_xy, goal_xy)
        cost = self.path_length(path) if path else None
        if cache and key is not None and len(self._cost_cache) < 4096:
            self._cost_cache[key] = cost
        return cost

    def path_is_clear(self, path, dynamic, from_index=0, look_ahead_m=None):
        """Is a previously planned path still legal given where peers are now?

        Used to trigger a replan *before* the speed governor has to brake,
        which is the difference between routing around a stopped robot and
        queueing behind it.
        """
        if not path or len(path) < 2:
            return True
        _extra, hard = self._rasterise([d for d in dynamic if d.hard])
        if not hard:
            return True
        grid = self.grid
        travelled = 0.0
        for a, b in zip(path[from_index:], path[from_index + 1:]):
            seg = math.hypot(b[0] - a[0], b[1] - a[1])
            steps = max(1, int(seg / grid.resolution))
            for k in range(steps + 1):
                f = k / steps
                wx = a[0] + (b[0] - a[0]) * f
                wy = a[1] + (b[1] - a[1]) * f
                gx, gy = grid.world_to_cell(wx, wy)
                if not grid.in_bounds(gx, gy):
                    continue
                if grid.index(gx, gy) in hard:
                    return False
            travelled += seg
            if look_ahead_m is not None and travelled > look_ahead_m:
                break
        return True
