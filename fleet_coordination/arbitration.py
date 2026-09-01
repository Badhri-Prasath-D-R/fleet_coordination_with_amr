#!/usr/bin/env python3
"""Right-of-way arbitration: aged priority, decided identically on both sides.

A fixed priority rank is what starved amr2 in the 09:16-09:18 logs -- 117
seconds elapsed and it moved one centimetre, because ``prio=2`` loses every
arbitration against ``prio=1``, forever, by construction. In a four-robot
fleet amr4 would effectively never move while the floor is busy.

Priority becomes a *cost* that waiting buys down:

    cost_i = base_priority_i - aging_rate * wait_i

Lower cost proceeds. Every robot broadcasts its own accumulated wait, so
both robots in a pair evaluate the same two numbers and reach opposite
conclusions without exchanging a single negotiation message. With
``aging_rate = 0.25`` the widest possible gap in a four-robot fleet
(amr1 vs amr4, 3.0 units) is erased after 12 s of waiting, so the 117 s
stall is not merely unlikely, it is structurally unreachable.

Two details make that safe in practice:

* **Deadband.** My wait is fresh; the peer's is up to one broadcast period
  old. Near a tie those two staleness errors can point opposite ways and
  both robots conclude they have right of way. Costs within ``deadband``
  are therefore treated as equal and settled by robot id, which is stale-
  proof. To shrink the error further, callers pass the value they last
  *broadcast* rather than their live counter, so both sides compare the
  same pair of published numbers.

* **Latching.** Aging keeps running during an encounter, so an unlatched
  rule can flip mid-pass -- the yielder's cost drops below the holder's,
  they swap roles, and both hesitate. A decision is therefore latched for
  the life of the encounter and only reconsidered once the pair is clear.
"""

__all__ = ["RightOfWay", "aged_cost"]


def aged_cost(base_priority, wait_s, aging_rate, max_credit):
    """Arbitration cost. Lower proceeds; waiting buys it down, but not forever."""
    credit = wait_s if wait_s < max_credit else max_credit
    return base_priority - aging_rate * credit


class RightOfWay:
    """Pairwise, symmetric, latched arbitration.

    ``decide`` is a pure function of the two broadcast (priority, wait)
    pairs plus the two robot ids, which is exactly what makes the peer
    running this same class arrive at the inverse answer.
    """

    def __init__(self, rid, base_priority, aging_rate=0.25, deadband=0.05,
                 max_wait_credit=30.0, latch_release_s=1.0):
        self.rid = rid
        self.base_priority = float(base_priority)
        self.aging_rate = float(aging_rate)
        self.deadband = float(deadband)
        self.max_wait_credit = float(max_wait_credit)
        self.latch_release_s = float(latch_release_s)
        self._latch = {}          # peer rid -> (i_have_row, last_conflict_time)

    # ------------------------------------------------------------------

    def _raw_decide(self, peer_rid, my_wait, peer_priority, peer_wait):
        mine = aged_cost(self.base_priority, my_wait,
                         self.aging_rate, self.max_wait_credit)
        theirs = aged_cost(peer_priority, peer_wait,
                           self.aging_rate, self.max_wait_credit)
        if abs(mine - theirs) > self.deadband:
            return mine < theirs
        return self.rid < peer_rid           # stale-proof deterministic tiebreak

    def decide(self, now, peer_rid, my_wait, peer_priority, peer_wait):
        """Do I proceed against this peer? Latched for the encounter's life."""
        latched = self._latch.get(peer_rid)
        if latched is not None and now - latched[1] <= self.latch_release_s:
            self._latch[peer_rid] = (latched[0], now)
            return latched[0]

        row = self._raw_decide(peer_rid, my_wait, peer_priority, peer_wait)
        self._latch[peer_rid] = (row, now)
        return row

    def expire(self, now):
        """Forget pairs that have been clear longer than the release window."""
        for rid in [r for r, (_, t) in self._latch.items()
                    if now - t > self.latch_release_s]:
            del self._latch[rid]

    def latched_with(self):
        return set(self._latch.keys())

    def my_cost(self, my_wait):
        return aged_cost(self.base_priority, my_wait,
                         self.aging_rate, self.max_wait_credit)
