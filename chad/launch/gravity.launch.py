"""Launch HEBI + robot_state_publisher + RViz + gravity node.

   The gravity node splines to the hardcoded waiting pose and holds there,
   sending the same gravity compensation (A*sin(shoulder)+B*cos(shoulder)) as
   touchtable for tuning A and B on this robot.

   Run: ros2 launch chad gravity.launch.py
"""

import os
from ament_index_python.packages import get_package_share_directory as pkgdir

from launch import LaunchDescription
from launch.actions import Shutdown
from launch_ros.actions import Node


def generate_launch_description():
    rvizcfg = os.path.join(pkgdir("chad"), "rviz/viewurdf.rviz")
    urdf = os.path.join(pkgdir("chad"), "urdf/robot.urdf")
    with open(urdf, "r") as f:
        robot_description = f.read()

    node_rviz = Node(
        name="rviz",
        package="rviz2",
        executable="rviz2",
        output="screen",
        arguments=["-d", rvizcfg],
        on_exit=Shutdown(),
    )
    node_robot_state_publisher = Node(
        name="robot_state_publisher",
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[{"robot_description": robot_description}],
    )
    node_hebi = Node(
        name="hebi",
        package="hebiros",
        executable="hebinode",
        output="screen",
        parameters=[
            {"family": "robotlab"},
            {"motors": ["base", "shoulder", "elbow", "wrist1", "wrist2", "gripper"]},
            {"joints": ["base", "shoulder", "elbow", "wrist1", "wrist2", "gripper"]},
        ],
        on_exit=Shutdown(),
    )
    node_gravity = Node(
        name="gravity",
        package="chad",
        executable="gravity",
        output="screen",
    )

    return LaunchDescription([
        node_rviz,
        node_robot_state_publisher,
        node_hebi,
        node_gravity,
    ])
