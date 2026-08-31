#!/usr/bin/env python3
"""
multi_robot_warehouse.launch.py

Spawns 3 bcr_bot AMRs into the SAME running Gazebo warehouse world, each in
its own ROS2 namespace (amr1, amr2, amr3), at different starting poses.

USAGE (two-terminal pattern, matching what already works reliably on WSL2):

  Terminal 1 - start Gazebo only, wait for the warehouse to fully load:
    ros2 launch gazebo_ros gazebo.launch.py \
        world:=$(ros2 pkg prefix bcr_bot)/share/bcr_bot/worlds/small_warehouse.sdf

  Terminal 2 - once warehouse is visible, spawn all 3 robots into it:
    ros2 launch bcr_bot multi_robot_warehouse.launch.py

Each robot's topics/tf will be cleanly namespaced, e.g.:
    /amr1/odom  /amr1/scan  /amr1/cmd_vel
    /amr2/odom  /amr2/scan  /amr2/cmd_vel
    /amr3/odom  /amr3/scan  /amr3/cmd_vel

This namespace separation is what makes P2P visible/provable: any node in
amr1's namespace can subscribe directly to /amr2/odom or /amr3/scan with
zero central broker - that's DDS discovery doing the work.
"""

from os.path import join

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import TimerAction
from launch.substitutions import Command, PythonExpression
from launch_ros.actions import Node, PushRosNamespace
from launch.actions import GroupAction


def make_robot_group(bcr_bot_path, xacro_path, namespace, x, y, yaw, spawn_delay):
    """
    Build one namespaced robot instance: robot_state_publisher + spawn_entity,
    wrapped in a PushRosNamespace group so ALL topics/tf for this robot are
    automatically prefixed (/amr1/..., /amr2/..., /amr3/...) without needing
    to manually remap every single topic.
    """

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        namespace=namespace,
        output='screen',
        parameters=[{
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
        namespace=namespace,
        output='screen',
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

    # Stagger spawns so we don't repeat the boot-time race we hit with a
    # single robot - each spawn_entity call gets gzserver already warm.
    return TimerAction(period=spawn_delay, actions=[group])


def generate_launch_description():
    bcr_bot_path = get_package_share_directory('bcr_bot')
    xacro_path = join(bcr_bot_path, 'urdf', 'bcr_bot.xacro')

    # Three starting poses spaced out in the warehouse aisles.
    # Adjust x/y once you've looked at small_warehouse.sdf's layout in Gazebo
    # so robots start in open floor space, not inside shelving.
    robots = [
        {"namespace": "amr1", "x": 0.0, "y": 0.0, "yaw": 0.0, "delay": 2.0},
        {"namespace": "amr2", "x": 2.5, "y": 0.0, "yaw": 0.0, "delay": 8.0},
        {"namespace": "amr3", "x": 5.0, "y": 0.0, "yaw": 0.0, "delay": 14.0},
    ]

    actions = []
    for r in robots:
        actions.append(
            make_robot_group(
                bcr_bot_path, xacro_path,
                r["namespace"], r["x"], r["y"], r["yaw"], r["delay"],
            )
        )

    return LaunchDescription(actions)