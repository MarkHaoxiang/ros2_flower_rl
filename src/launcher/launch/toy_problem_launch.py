from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    # Launch Arguments
    experiment_dataset_value = LaunchConfiguration("experiment_dataset")
    n_partitions_value = LaunchConfiguration("n_partitions")

    experiment_dataset_launch_arg = DeclareLaunchArgument(
        "experiment_dataset",
        default_value="mnist"
    )
    n_partitions_value_launch_arg = DeclareLaunchArgument(
        "n_partitions",
        default_value='10'
    )

    #return LaunchDescription(
    #    [
    #        Node(
    #            package="flower_subscriber",
    #            namespace="client_1",
    #            executable="flower_subscriber",
    #            name="sim",
    #        ),
    #        Node(
    #            package="flower_subscriber",
    #            namespace="client_2",
    #            executable="flower_subscriber",
    #            name="sim",
    #        ),
    #        # Add experiment data publishers as well
    #        # Node(
    #        #     package="turtlesim",
    #        #     executable="mimic",
    #        #     name="mimic",
    #        #     remappings=[
    #        #         ("/input/pose", "/turtlesim1/turtle1/pose"),
    #        #         ("/output/cmd_vel", "/turtlesim2/turtle1/cmd_vel"),
    #        #     ],
    #        # ),
    #    ]
    #)
    return LaunchDescription([])
