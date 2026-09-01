#!/usr/bin/env python3
"""Bring up the full decentralized coordination stack for the AMR fleet.

Run this AFTER Gazebo is up and the robots have been spawned by
``multi_robot_warehouse.launch.py``.

Per robot it starts three nodes, all in that robot namespace:

    fleet_task_allocator   decides what this robot works on   -> nav_goal
    fleet_navigator        decides which way it goes          -> cmd_vel_nav
    fleet_collision_node   decides whether and how fast       -> cmd_vel

and fleet-wide it starts one monitor (read-only) and one dispatcher (which
injects work and then has nothing more to do with it).

Nothing here is a fleet manager. The three per-robot nodes talk to their
peers over four global topics -- /fleet/state, /fleet/market,
/fleet/task_request and /fleet/events -- and to nothing else. Kill any robot
and the others notice its heartbeat stop and reassign its work; start a
fifth and it joins the auction on its next tick.

Usage::

    ros2 launch bcr_bot fleet_coordination.launch.py
    ros2 launch bcr_bot fleet_coordination.launch.py robots:=amr1,amr2,amr3,amr4
    ros2 launch bcr_bot fleet_coordination.launch.py dispatch_mode:=random

If you would rather drive with Nav2 than with this package planner, set
``use_navigator:=false`` and remap your Nav2 controller_server output to
``cmd_vel_nav``; the collision node sits in exactly the same place either way.
"""

from os.path import join

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, OpaqueFunction,
                            TimerAction)
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# base_priority is only a tiebreak seed now. Aging (see arbitration.py) means
# it no longer decides every encounter, so amr4 does not starve the way amr2
# did in the original logs.
BASE_PRIORITY = {"amr1": 1.0, "amr2": 2.0, "amr3": 3.0, "amr4": 4.0}


def build(context, *_args, **_kwargs):
    robots = [r.strip() for r in
              LaunchConfiguration("robots").perform(context).split(",")
              if r.strip()]
    params_file = LaunchConfiguration("params_file").perform(context)
    map_yaml = LaunchConfiguration("map_yaml").perform(context)
    tasks_file = LaunchConfiguration("tasks_file").perform(context)
    log_dir = LaunchConfiguration("log_dir").perform(context)
    use_navigator = LaunchConfiguration("use_navigator").perform(context)
    use_allocator = LaunchConfiguration("use_allocator").perform(context)

    actions = []

    for index, rid in enumerate(robots):
        common = {
            "robot_id": rid,
            "use_sim_time": True,
        }

        collision = Node(
            package="bcr_bot",
            executable="fleet_collision_node.py",
            name="fleet_collision_node",
            namespace=rid,
            output="screen",
            parameters=[params_file, dict(
                common,
                base_priority=BASE_PRIORITY.get(rid, float(index + 1)),
                log_file=join(log_dir, "%s_fleet.log" % rid),
            )],
            remappings=[
                # /fleet/* stay absolute on purpose: they are the shared bus.
                ("odom", "/%s/odom" % rid),
                ("scan", "/%s/scan" % rid),
                ("cmd_vel", "/%s/cmd_vel" % rid),
                ("cmd_vel_nav", "/%s/cmd_vel_nav" % rid),
            ],
        )

        navigator = Node(
            package="bcr_bot",
            executable="fleet_navigator.py",
            name="fleet_navigator",
            namespace=rid,
            output="screen",
            condition=IfCondition(use_navigator),
            parameters=[params_file, dict(
                common,
                map_yaml=map_yaml,
                log_file=join(log_dir, "%s_fleet.log" % rid),
            )],
            remappings=[
                ("odom", "/%s/odom" % rid),
                ("cmd_vel_nav", "/%s/cmd_vel_nav" % rid),
            ],
        )

        allocator = Node(
            package="bcr_bot",
            executable="fleet_task_allocator.py",
            name="fleet_task_allocator",
            namespace=rid,
            output="screen",
            condition=IfCondition(use_allocator),
            parameters=[params_file, dict(
                common,
                map_yaml=map_yaml,
                log_file=join(log_dir, "%s_fleet.log" % rid),
            )],
            remappings=[("odom", "/%s/odom" % rid)],
        )

        # The robots publish ground-truth odometry in world coordinates
        # (odometry_source:=world in the spawn launch), so map -> <rid>/odom
        # really is the identity. Publishing it gives RViz one coherent tree
        # to draw all four robots and the shared markers in.
        map_tf = Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="map_to_%s_odom" % rid,
            output="log",
            arguments=["0", "0", "0", "0", "0", "0", "map", "%s/odom" % rid],
            parameters=[{"use_sim_time": True}],
        )

        # Each Node already carries namespace=rid, so there is deliberately
        # no PushRosNamespace here -- doing both would yield /amr1/amr1/...
        actions.extend([collision, navigator, allocator, map_tf])

    monitor = Node(
        package="bcr_bot",
        executable="fleet_monitor.py",
        name="fleet_monitor",
        output="screen",
        condition=IfCondition(LaunchConfiguration("use_monitor")),
        parameters=[params_file, {
            "use_sim_time": True,
            "event_log": join(log_dir, "fleet_events.log"),
        }],
    )

    dispatcher = Node(
        package="bcr_bot",
        executable="fleet_task_dispatcher.py",
        name="fleet_task_dispatcher",
        output="screen",
        condition=IfCondition(LaunchConfiguration("use_dispatcher")),
        parameters=[params_file, {
            "use_sim_time": True,
            "tasks_file": tasks_file,
            "mode": LaunchConfiguration("dispatch_mode"),
        }],
    )

    # Give every robot a moment to appear on /fleet/state before any work is
    # auctioned, so the first task is priced against the whole fleet rather
    # than whoever happened to boot first.
    actions.append(monitor)
    actions.append(TimerAction(period=3.0, actions=[dispatcher]))
    return actions


def generate_launch_description():
    pkg = get_package_share_directory("bcr_bot")

    return LaunchDescription([
        DeclareLaunchArgument(
            "robots", default_value="amr1,amr2,amr3",
            description="Comma-separated robot namespaces to coordinate."),
        DeclareLaunchArgument(
            "params_file",
            default_value=join(pkg, "config", "fleet_coordination.yaml"),
            description="Shared parameter file for every fleet node."),
        DeclareLaunchArgument(
            "map_yaml", default_value=join(pkg, "config", "bcr_map.yaml"),
            description="Occupancy map used for routing and bid pricing."),
        DeclareLaunchArgument(
            "tasks_file", default_value=join(pkg, "config", "fleet_tasks.yaml"),
            description="Station list and scripted workload."),
        DeclareLaunchArgument(
            "log_dir", default_value="/tmp/fleet_logs",
            description="Where each robot writes its structured log."),
        DeclareLaunchArgument(
            "dispatch_mode", default_value="scripted",
            description="scripted (config/fleet_tasks.yaml) or random."),
        DeclareLaunchArgument("use_monitor", default_value="true"),
        DeclareLaunchArgument("use_dispatcher", default_value="true"),
        DeclareLaunchArgument("use_navigator", default_value="true"),
        DeclareLaunchArgument("use_allocator", default_value="true"),
        OpaqueFunction(function=build),
    ])
