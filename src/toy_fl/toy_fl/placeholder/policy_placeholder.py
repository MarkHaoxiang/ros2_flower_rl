import rclpy
from rclpy.node import Node
import gymnasium as gym
import numpy as np

import ml_interfaces.msg as msg
import ml_interfaces.srv as srv
from ml_interfaces_py import FloatTensor


class PolicyPlaceholder(Node):
    """Publishes random actions as an utility to help test gym controller"""

    def __init__(self, space: gym.spaces.Space):
        super().__init__("actor_placeholder")
        self.srv = self.create_service(
            srv.PolicyService, "policy", self.policy_callback
        )
        self.space = space

    def policy_callback(
        self, request: srv.PolicyService.Request, response: srv.PolicyService.Response
    ):
        action = FloatTensor.build(np.array(self.space.sample()))
        response.a = action.pack()
        self.get_logger().info(f"Completed policy calculation for {str(request)}")
        return response


def main(args=None):
    rclpy.init(args=args)
    env = gym.make("CartPole")
    node = PolicyPlaceholder(space=env.action_space)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
