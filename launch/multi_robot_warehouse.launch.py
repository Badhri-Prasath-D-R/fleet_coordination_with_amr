#!/usr/bin/env python3
"""
multi_robot_warehouse.launch.py

Spawns N bcr_bot AMRs into the SAME running Gazebo warehouse world, each in
its own ROS2 namespace (amr1, amr2, ...), at different starting poses.

USAGE (two-terminal pattern, matching what already works reliably on WSL2):

  Terminal 1 - start Gazebo only, wait for the warehouse to fully load:
    ros2 launch gazebo_ros gazebo.launch.py \
        world:=$(ros2 pkg prefix bcr_bot)/share/bcr_bot/worlds/small_warehouse.sdf

  Terminal 2 - once the warehouse is visible, spawn the robots into it:
    ros2 launch bcr_bot multi_robot_warehouse.launch.py
    ros2 launch bcr_bot multi_robot_warehouse.launch.py robots:=amr1,amr2,amr3,amr4

  Terminal 3 - bring up the coordination stack:
    ros2 launch bcr_bot fleet_coordination.launch.py robots:=amr1,amr2,amr3

Each robot topics/tf are cleanly namespaced, e.g.:
    /amr1/odom  /amr1/scan  /amr1/cmd_vel
    /amr2/odom  /amr2/scan  /amr2/cmd_vel

That namespace separation is what makes the peer-to-peer layer visible and
provable: any node in amr1 namespace can subscribe directly to /amr2/odom
with zero central broker - that is DDS discovery doing the work.

Two things here are load-bearing for the coordination stack downstream:

  * ``odometry_source:=world`` makes /amrN/odom absolute ground truth in the
    world frame. That is why the fleet nodes can plan on bcr_map.pgm without
    AMCL, and why ``map -> amrN/odom`` is published as the identity in
    fleet_coordination.launch.py.

  * The starting poses below were checked against bcr_map.pgm and all have
    at least 1.5 m of clearance. The previous poses (0,0), (2.5,0) and
    (5.0,0) put two of the three robots inside the racking: (2.5,0) has
    0.33 m of clearance and (5.0,0) has 0.30 m, and a 0.64 m wide chassis
    cannot physically occupy either. Run scripts/verify_fleet_config.py
    after changing them.
"""

from os.path import join

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, GroupAction, OpaqueFunction,
                            TimerAction)
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node, PushRosNamespace

# x, y, yaw. Clearances at these points, measured off config/bcr_map.pgm:
# amr1 2.16 m, amr2 1.80 m, amr3 1.50 m, amr4 2.10 m. They are also at least
# 3 m apart, so nobody starts inside a peer conflict radius.
START_POSES = {
    "amr1": (-4.50, -7.75, 0.0),
    "amr2": (-5.00, -3.50, 0.0),
    "amr3": (-2.00, -0.25, 0.0),
    "amr4": (-4.75, 1.25, 0.0),
}


def make_robot_group(xacro_path, namespace, x, y, yaw, spawn_delay):
    """
    Build one namespaced robot instance: robot_state_publisher + spawn_entity,
    wrapped in a PushRosNamespace group so ALL topics/tf for this robot are
    automatically prefixed (/amr1/..., /amr2/...) without needing to manually
    remap every single topic.
    """

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'robot_description': Command([
                'xacro ', xacro_path,
                ' camera_enabled:=', 'true',
                ' stereo_camera_enabled:=', 'false',
                ' two_d_lidar_enabled:=', 'true',
                ' sim_gazebo:=', 'true',
                ' odometry_source:=', 'world',
                ' robot_namespace:=', namespace,
            ]),
            'frame_prefix': f'{namespace}/',
        }],
    )

    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        output='screen',
        parameters=[{'use_sim_time': True}],
        arguments=[
            '-topic', 'robot_description',
            '-entity', f'{namespace}_robot',
            '-robot_namespace', namespace,
            '-z', '0.28',
            '-x', str(x),
            '-y', str(y),
            '-Y', str(yaw),
        ],
    )

    group = GroupAction([
        PushRosNamespace(namespace),
        robot_state_publisher,
        spawn_entity,
    ])

    # Stagger spawns so we do not repeat the boot-time race we hit with a
    # single robot - each spawn_entity call gets gzserver already warm.
    return TimerAction(period=spawn_delay, actions=[group])


def build(context, *_args, **_kwargs):
    bcr_bot_path = get_package_share_directory('bcr_bot')
    xacro_path = join(bcr_bot_path, 'urdf', 'bcr_bot.xacro')

    robots = [r.strip() for r in
              LaunchConfiguration('robots').perform(context).split(',')
              if r.strip()]
    first_delay = float(LaunchConfiguration('first_delay').perform(context))
    stagger = float(LaunchConfiguration('stagger').perform(context))

    actions = []
    for i, namespace in enumerate(robots):
        if namespace not in START_POSES:
            raise RuntimeError(
                "no start pose defined for %r -- add one to START_POSES in "
                "multi_robot_warehouse.launch.py and re-check it with "
                "scripts/verify_fleet_config.py" % namespace)
        x, y, yaw = START_POSES[namespace]
        actions.append(make_robot_group(
            xacro_path, namespace, x, y, yaw, first_delay + i * stagger))
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'robots', default_value='amr1,amr2,amr3',
            description='Comma-separated robot namespaces to spawn.'),
        DeclareLaunchArgument(
            'first_delay', default_value='2.0',
            description='Seconds to wait before spawning the first robot.'),
        DeclareLaunchArgument(
            'stagger', default_value='6.0',
            description='Seconds between consecutive spawns.'),
        OpaqueFunction(function=build),
    ])
