import rclpy
from rclpy.node import Node
import gymnasium as gym

import ml_interfaces.msg as msg
from ml_interfaces_py import Transition


class TransitionPublisherPlaceholder(Node):
    """Publishes random transitions as an utility to help test replay_buffer

    Equivalent to a simple version of early_start: randomly exploring in the environment
    """

    def __init__(self, topic: str = "observations"):
        super().__init__("transition_publisher")
        self.get_logger().info(f"Building publisher {self.get_fully_qualified_name()}")

        self.env = gym.make("CartPole")
        self.s_0, _ = self.env.reset()

        # Build publisher
        self._publisher = self.create_publisher(msg.Transition, topic, 10)
        self.timer = self.create_timer(
            0.5,
            self.timer_callback,
        )

    def timer_callback(self):
        a = self.env.action_space.sample()
        s_1, r, d, t, _ = self.env.step(a)

        # Pack and send
        msg = Transition.build((self.s_0, a, r, s_1, d))
        self._publisher.publish(msg.pack())

        # Reset env if needed
        if d or t:
            self.s_0, _ = self.env.reset()
        else:
            self.s_0 = s_1


def main(args=None):
    rclpy.init(args=args)
    node = TransitionPublisherPlaceholder()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
