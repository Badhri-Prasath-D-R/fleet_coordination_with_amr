"""Decentralized multi-AMR fleet coordination.

Four cooperating layers, each in its own node, sharing one JSON bus:

    fleet_task_allocator   decentralized auction        -> who does what
    fleet_navigator        A* + pure pursuit + reroute  -> which way
    fleet_collision_node   CPA speed governor           -> whether / how fast
    fleet_monitor          read-only aggregation        -> what happened

Nothing here is a central server. Every node runs once per robot, publishes
its own state, and reasons locally from what its peers broadcast. Decisions
that need agreement (right of way, auction winners) use rules that are
*symmetric*: every robot feeding the same inputs into the same function gets
the same answer, so no negotiation round-trip is required.
"""

__all__ = [
    "arbitration",
    "bus",
    "flog",
    "geometry",
    "gridmap",
    "planner",
    "tasks",
]
