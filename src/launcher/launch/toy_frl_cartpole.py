from launch import LaunchDescription
from launch.actions import OpaqueFunction, RegisterEventHandler
from launch.event_handlers.on_process_start import OnProcessStart
from launch_ros.actions import Node


def generate_launch_description():
    N_CLIENTS = 5

    memory_nodes = []
    policy_nodes = []
    controller_nodes = []
    train_nodes = []
    for i in range(N_CLIENTS):
        namespace = f"client_{i}"
        memory_node = Node(
            package="replay_buffer",
            executable="transition_memory",
            name="replay_buffer",
            namespace=namespace
        )

        policy_node = Node(
            package="toy_frl",
            executable="dqn_actor",
            name="policy_provider",
            namespace=namespace
        )
        _controller_node = Node(
            package="toy_frl",
            executable="gym_controller",
            name="gym_controller",
            parameters=[
                { "publish_frequency": 10.0 }
            ],
            namespace=namespace
        )
        controller_node = RegisterEventHandler(
            event_handler=OnProcessStart(
            target_action=policy_node,
            on_start=OpaqueFunction(function=lambda _: [ _controller_node ])
        ))
        _train_node = Node(
            package="toy_frl",
            executable="dqn_client",
            name="flower_client",
            namespace=namespace
        )
        train_node = RegisterEventHandler(
            event_handler=OnProcessStart(
            target_action=memory_node,
            on_start=OpaqueFunction(function=lambda _: [ _train_node ])
        ))
        memory_nodes.append(memory_node)
        policy_nodes.append(policy_node)
        controller_nodes.append(controller_node)
        train_nodes.append(train_node)

    return LaunchDescription([
        memory_nodes, policy_nodes, controller_nodes, train_nodes
    ])

