import os
from rospkg import get_ros_home

from launch import LaunchDescription, LaunchContext
from launch.actions import DeclareLaunchArgument, OpaqueFunction, RegisterEventHandler
from launch.event_handlers.on_process_exit import OnProcessExit
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    EXPERIMENT_DATASET = "experiment_dataset"
    N_PARTITIONS = "n_partitions"

    # Downloader
    global_dataset_dir = os.path.join(get_ros_home(), "fl_data")
    experiment_dataset_value = LaunchConfiguration(EXPERIMENT_DATASET)
    n_partitions_value = LaunchConfiguration(N_PARTITIONS)
    experiment_dataset_launch_arg = DeclareLaunchArgument(
        EXPERIMENT_DATASET, default_value="mnist"
    )
    n_partitions_value_launch_arg = DeclareLaunchArgument(
        N_PARTITIONS, default_value="10"
    )
    downloader_node = Node(
        package="toy_fl_publisher",
        executable="downloader",
        name="downloader",
        parameters=[
            {
                EXPERIMENT_DATASET: experiment_dataset_value,
                N_PARTITIONS: n_partitions_value,
            }
        ],
    )

    # Publishers
    def generate_publisher_nodes(context: LaunchContext):
        n = int(context.launch_configurations[N_PARTITIONS])
        experiment_dataset = str(context.launch_configurations[EXPERIMENT_DATASET])
        dataset_dir = os.path.join(global_dataset_dir, experiment_dataset)
        publishers = []
        for i in range(n):
            publishers.append(
                Node(
                    package="toy_fl_publisher",
                    executable="publisher",
                    namespace=f"client_{i}",
                    parameters=[
                        {
                            "dataset_dir": os.path.join(
                                dataset_dir, f"{experiment_dataset}_{i}.hf"
                            )
                        }
                    ],
                )
            )
        return publishers

    # Publishers
    def generate_subscriber_nodes(context: LaunchContext):
        n = int(context.launch_configurations[N_PARTITIONS])
        subscribers = []
        for i in range(n):
            subscribers.append(
                Node(
                    package="flower_subscriber",
                    executable="subscriber",
                    namespace=f"client_{i}",
                )
            )
        return subscribers

    publisher_nodes = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=downloader_node,
            on_exit=OpaqueFunction(function=generate_publisher_nodes),
        )
    )

    subscriber_nodes = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=downloader_node,
            on_exit=OpaqueFunction(function=generate_subscriber_nodes),
        )
    )

    # Return Launch Description
    return LaunchDescription(
        [
            # Downloader
            experiment_dataset_launch_arg,
            n_partitions_value_launch_arg,
            downloader_node,
            publisher_nodes,
            subscriber_nodes,
        ]
    )
