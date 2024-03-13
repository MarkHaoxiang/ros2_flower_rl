import os
from rospkg import get_ros_home

from launch import LaunchDescription, LaunchContext
from launch.actions import DeclareLaunchArgument, OpaqueFunction, RegisterEventHandler
from launch.event_handlers.on_process_start import OnProcessStart
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    memory_node = Node(
        package="replay_buffer",
        executable="transition_memory",
        name="replay_buffer"
    )

    policy_node = Node(
        package="toy_frl",
        executable="dqn_actor",
        name="policy_provider"
    )


    _controller_node = Node(
        package="toy_frl",
        executable="gym_controller",
        name="gym_controller",
        parameters=[
            { "publish_frequency": 10.0 }
        ]
    )

    controller_node = RegisterEventHandler(
        event_handler=OnProcessStart(
        target_action=policy_node,
        on_start=OpaqueFunction(function=lambda _: [ _controller_node ])
    ))

    _train_node = Node(
        package="toy_frl",
        executable="dqn_client",
        name="flower_client"
    )

    train_node = RegisterEventHandler(
        event_handler=OnProcessStart(
        target_action=memory_node,
        on_start=OpaqueFunction(function=lambda _: [ _train_node ])
    ))

    return LaunchDescription([
        memory_node, policy_node, controller_node, train_node
    ])

