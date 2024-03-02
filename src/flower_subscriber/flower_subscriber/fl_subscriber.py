import rclpy
from rclpy.node import Node


class FlowerSubscriber(Node):
    """
    ROS node responsible for feeding data to flower client, in which the training occurs
    """

    def __init__(self, flower_hook_fn):
        super().__init__("flower_subscriber")
        self.subscriptions = self.create_subscription(
            msg_type="uint8[]",  # Subject to changes
            topic="training_data",
            callback=self.listener_callback,
            raw=True,  # Store incoming data as raw bytes, could be useful if we are actuall sending
            # floats
        )
        self.subscription
        self.external_callback = flower_hook_fn

    def listener_callback(self, msg):
        # TODO: link this to be handled in the flower client
        # TODO: Part 1: write aggregate logic for training data
        # TODO: Part 2: send training data to flower client for training
        raise NotImplementedError
