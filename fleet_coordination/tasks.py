#!/usr/bin/env python3
"""Decentralized task allocation: a contract-net auction with no auctioneer.

Every robot runs an identical copy of :class:`Market`. Work is announced on
a shared topic, everyone who has capacity prices it and broadcasts a bid,
and when the bid window closes **every robot independently computes the same
winner** from the same set of bids. There is no allocator process to elect,
to fail over, or to bottleneck on.

Three things have to be true for that to actually work rather than merely
sound decentralized, and each one is handled explicitly below:

1. *Determinism.* Winner selection is ``min`` over ``(cost, robot_id)`` with
   the cost rounded before comparison, so two robots cannot disagree because
   of a float that differed in the last bit.

2. *Convergence under message loss.* A robot that missed a bid will compute
   a different winner. The winner therefore also asserts its claim with an
   AWARD message, and every robot resolves a contested claim with the same
   ``(cost, robot_id)`` comparison. Whoever loses that comparison drops the
   task -- including a robot that had already started it.

3. *Fault tolerance.* An owner that stops broadcasting has its task
   reopened. Exactly one peer does the reopening -- the lowest-id live
   robot -- so a dead robot produces one re-announcement rather than a storm
   of three.
"""

import math

__all__ = [
    "Market",
    "Task",
    "bid_cost",
    "msg_announce",
    "msg_award",
    "msg_bid",
    "msg_done",
    "msg_progress",
    "msg_release",
]

# Task lifecycle.
OPEN = "OPEN"            # announced, bids being collected
ASSIGNED = "ASSIGNED"    # a winner is known, not yet started
ACTIVE = "ACTIVE"        # the owner reports it is executing
DONE = "DONE"
FAILED = "FAILED"        # nobody could route to it; retired

TERMINAL = (DONE, FAILED)


class Task:
    """One transport job: go to ``pickup``, then carry to ``dropoff``."""

    __slots__ = ("tid", "label", "pickup", "dropoff", "priority", "created",
                 "state", "owner", "owner_cost", "announced", "bids",
                 "attempts", "excluded", "last_change")

    def __init__(self, tid, pickup, dropoff, label="", priority=1.0,
                 created=0.0):
        self.tid = tid
        self.label = label or tid
        self.pickup = (float(pickup[0]), float(pickup[1]))
        self.dropoff = (float(dropoff[0]), float(dropoff[1]))
        self.priority = float(priority)
        self.created = float(created)
        self.state = OPEN
        self.owner = None
        self.owner_cost = None
        self.announced = float(created)
        self.bids = {}
        self.attempts = 0
        self.excluded = set()      # robots that tried and could not route
        self.last_change = float(created)

    # ------------------------------------------------------ serialisation

    def to_dict(self):
        return {
            "tid": self.tid,
            "label": self.label,
            "pickup": [round(self.pickup[0], 3), round(self.pickup[1], 3)],
            "dropoff": [round(self.dropoff[0], 3), round(self.dropoff[1], 3)],
            "priority": self.priority,
            "created": round(self.created, 3),
            "attempts": self.attempts,
            "excluded": sorted(self.excluded),
        }

    @staticmethod
    def from_dict(d):
        t = Task(d["tid"], d["pickup"], d["dropoff"],
                 label=d.get("label", ""),
                 priority=float(d.get("priority", 1.0)),
                 created=float(d.get("created", 0.0)))
        t.attempts = int(d.get("attempts", 0))
        t.excluded = set(d.get("excluded", []))
        return t

    def __repr__(self):
        return ("<Task %s %s owner=%s %s->%s>"
                % (self.tid, self.state, self.owner, self.pickup, self.dropoff))


# ------------------------------------------------------------- messages

def msg_announce(task, sender, t):
    return {"type": "ANNOUNCE", "from": sender, "t": round(t, 3),
            "task": task.to_dict()}


def msg_bid(tid, sender, cost, t):
    return {"type": "BID", "from": sender, "t": round(t, 3),
            "tid": tid, "cost": round(float(cost), 3)}


def msg_award(tid, winner, cost, sender, t):
    return {"type": "AWARD", "from": sender, "t": round(t, 3),
            "tid": tid, "winner": winner, "cost": round(float(cost), 3)}


def msg_progress(tid, sender, phase, t):
    return {"type": "PROGRESS", "from": sender, "t": round(t, 3),
            "tid": tid, "phase": phase}


def msg_done(tid, sender, t):
    return {"type": "DONE", "from": sender, "t": round(t, 3), "tid": tid}


def msg_release(tid, sender, reason, t):
    return {"type": "RELEASE", "from": sender, "t": round(t, 3),
            "tid": tid, "reason": reason}


# ------------------------------------------------------------- bidding

def bid_cost(route_len_to_pickup, route_len_pickup_to_dropoff, nominal_speed,
             queue_len, busy_remaining_m=0.0, congestion=0.0,
             queue_penalty_s=8.0, congestion_penalty_s=6.0):
    """Estimated seconds until this robot would *finish* the task.

    Bidding in time-to-completion rather than in distance-to-pickup is what
    stops the nearest robot from hoarding: a robot standing on the pickup
    but with two jobs already queued honestly bids worse than one that is
    six metres away and idle, and the fleet balances itself without any
    central load balancer.

    Returns ``None`` when either leg is unroutable, which is a bid the robot
    simply does not place.
    """
    if route_len_to_pickup is None or route_len_pickup_to_dropoff is None:
        return None
    speed = max(0.05, float(nominal_speed))
    seconds = (float(busy_remaining_m) + float(route_len_to_pickup)
               + float(route_len_pickup_to_dropoff)) / speed
    seconds += queue_penalty_s * max(0, int(queue_len))
    seconds += congestion_penalty_s * float(congestion)
    return seconds


# --------------------------------------------------------------- market

class Market:
    """Every robot view of the shared task board.

    Identical inputs produce identical outputs on every robot, which is the
    entire trick: agreement is reached by computation rather than by
    conversation.
    """

    def __init__(self, rid, bid_window_s=1.0, owner_timeout_s=4.0,
                 reopen_period_s=5.0, max_attempts=4, assign_timeout_s=8.0):
        self.rid = rid
        self.bid_window = float(bid_window_s)
        self.owner_timeout = float(owner_timeout_s)
        self.reopen_period = float(reopen_period_s)
        self.max_attempts = int(max_attempts)
        self.assign_timeout = float(assign_timeout_s)
        self.tasks = {}
        self.settled = set()          # tids whose current round we have closed

    # ------------------------------------------------------------ ingest

    def announce(self, task, now):
        """Record a new or reopened task. Returns True if this is new to us."""
        existing = self.tasks.get(task.tid)
        if existing is None:
            task.announced = now
            task.last_change = now
            self.tasks[task.tid] = task
            self.settled.discard(task.tid)
            return True

        if existing.state in TERMINAL:
            return False
        # A reopen: same tid, higher attempt count. Reset the round.
        if task.attempts > existing.attempts:
            existing.attempts = task.attempts
            existing.excluded |= task.excluded
            existing.state = OPEN
            existing.owner = None
            existing.owner_cost = None
            existing.bids = {}
            existing.announced = now
            existing.last_change = now
            self.settled.discard(task.tid)
            return True
        return False

    def record_bid(self, tid, rid, cost, now):
        task = self.tasks.get(tid)
        if task is None or task.state != OPEN:
            return
        task.bids[rid] = float(cost)
        task.last_change = now

    def apply_award(self, tid, winner, cost, now):
        """Adopt or reject a claim. Returns True if our own view changed.

        Both sides of a contested claim run this same comparison, so the
        disagreement is resolved in one message round with no follow-up.
        """
        task = self.tasks.get(tid)
        if task is None or task.state in TERMINAL:
            return False

        cost = float(cost)
        if task.owner is None:
            task.owner, task.owner_cost = winner, cost
            task.state = ASSIGNED
            task.last_change = now
            self.settled.add(tid)
            return True

        if task.owner == winner:
            task.owner_cost = min(task.owner_cost, cost)
            return False

        incumbent = (round(task.owner_cost, 3), task.owner)
        challenger = (round(cost, 3), winner)
        if challenger < incumbent:
            task.owner, task.owner_cost = winner, cost
            task.state = ASSIGNED
            task.last_change = now
            self.settled.add(tid)
            return True
        return False

    def apply_progress(self, tid, rid, phase, now):
        task = self.tasks.get(tid)
        if task is None or task.owner != rid:
            return
        task.state = ACTIVE
        task.last_change = now

    def apply_done(self, tid, rid, now):
        task = self.tasks.get(tid)
        if task is None:
            return
        task.state = DONE
        task.owner = rid
        task.last_change = now

    def apply_release(self, tid, rid, now, exclude=True):
        """An owner gave a task back. Reopen it, optionally barring the quitter."""
        task = self.tasks.get(tid)
        if task is None or task.state in TERMINAL:
            return
        if exclude:
            task.excluded.add(rid)
        task.state = OPEN
        task.owner = None
        task.owner_cost = None
        task.bids = {}
        task.attempts += 1
        task.announced = now
        task.last_change = now
        self.settled.discard(tid)

    # ---------------------------------------------------------- decisions

    def due_for_settlement(self, now):
        """Tasks whose bid window has closed and that we have not settled."""
        return [t for t in self.tasks.values()
                if t.state == OPEN and t.tid not in self.settled
                and now - t.announced >= self.bid_window]

    @staticmethod
    def winner_of(task):
        """Deterministic winner: cheapest bid, ties broken by robot id.

        Costs are rounded before comparison so that two robots holding the
        same bids can never order them differently.
        """
        if not task.bids:
            return None, None
        rid = min(task.bids, key=lambda r: (round(task.bids[r], 3), r))
        return rid, task.bids[rid]

    def i_am_lowest_live(self, live_ids):
        """Am I the peer responsible for housekeeping this tick?

        Reopening a dead robot tasks is a fleet-wide duty that must be
        performed exactly once. Rather than electing a leader, every robot
        asks whether it is the lowest id currently alive -- which is a
        function of state everyone already has.
        """
        ids = set(live_ids)
        ids.add(self.rid)
        return self.rid == min(ids)

    def orphaned(self, live_ids, now):
        """Tasks that need pushing: dead owner, no bids, or never started.

        The third case is the subtle one. Every robot settles its own
        auction, so a robot that missed a bid can conclude that peer X won
        and sit waiting for X to get on with it -- while X, having seen the
        full bid set, concluded it lost. Nobody starts. A task that is
        ASSIGNED but silent past ``assign_timeout`` is therefore reopened,
        which converts a rare disagreement into a few seconds of delay
        instead of a task that is never done.
        """
        live = set(live_ids)
        live.add(self.rid)
        out = []
        for t in self.tasks.values():
            if t.state in TERMINAL:
                continue
            if t.owner is not None and t.owner not in live:
                out.append(t)
            elif (t.state == ASSIGNED
                  and now - t.last_change >= self.assign_timeout):
                out.append(t)
            elif (t.state == OPEN and t.tid in self.settled
                  and not t.bids
                  and now - t.announced >= self.reopen_period):
                out.append(t)
        return out

    def reopen(self, task, now, reason=""):
        """Prepare a task for another auction round. False once it is exhausted."""
        task.attempts += 1
        if task.attempts > self.max_attempts:
            task.state = FAILED
            task.last_change = now
            return False
        task.state = OPEN
        task.owner = None
        task.owner_cost = None
        task.bids = {}
        task.announced = now
        task.last_change = now
        self.settled.discard(task.tid)
        return True

    def mine(self):
        return [t for t in self.tasks.values()
                if t.owner == self.rid and t.state not in TERMINAL]

    def open_tasks(self):
        return [t for t in self.tasks.values() if t.state == OPEN]

    def summary(self):
        counts = {}
        for t in self.tasks.values():
            counts[t.state] = counts.get(t.state, 0) + 1
        return counts

    def forget_completed(self, now, keep_s=120.0):
        """Retire terminal tasks so the board does not grow without bound."""
        for tid in [t.tid for t in self.tasks.values()
                    if t.state in TERMINAL and now - t.last_change > keep_s]:
            del self.tasks[tid]
            self.settled.discard(tid)


def euclid(a, b):
    return math.hypot(b[0] - a[0], b[1] - a[1])
