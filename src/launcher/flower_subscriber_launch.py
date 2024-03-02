from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            Node(
                package="flower_subscriber",
                namespace="client_1",
                executable="flower_subscriber",
                name="sim",
            ),
            Node(
                package="flower_subscriber",
                namespace="client_2",
                executable="flower_subscriber",
                name="sim",
            ),
            # Add experiment data publishers as well
            # Node(
            #     package="turtlesim",
            #     executable="mimic",
            #     name="mimic",
            #     remappings=[
            #         ("/input/pose", "/turtlesim1/turtle1/pose"),
            #         ("/output/cmd_vel", "/turtlesim2/turtle1/cmd_vel"),
            #     ],
            # ),
        ]
    )
