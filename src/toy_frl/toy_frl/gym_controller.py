# ruff: noqa: F401

import gymnasium as gym
import ml_interfaces.msg as msg
import ml_interfaces.srv as srv
import rclpy
from ml_interfaces_py import FloatTensor, Transition
from rcl_interfaces.msg import SetParametersResult
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from toy_frl.dqn.common import POLICY_SERVICE, POLICY_UPDATE_TOPIC

class GymController(Node):
    """A robot controller simulating an Gymnasium RL Environment"""

    def __init__(self, env_name: str):
        super().__init__("gymnasium_controller")
        self.get_logger().info(f"Building controller {self.get_fully_qualified_name()}")

        self.env = gym.make(env_name)
        self.s_0, _ = self.env.reset()

        # ReentrantCallback group enables client and timer to execute concurrently
        self.cb_group = ReentrantCallbackGroup()

        # Build Policy client
        self.policy_client = self.create_client(
            srv_type=srv.PolicyService, srv_name=POLICY_SERVICE, callback_group=self.cb_group
        )
        while not self.policy_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Policy service not available... waiting")

        # Transition publisher
        self.publisher = self.create_publisher(
            msg_type=msg.Transition, topic="observations", qos_profile=10
        )
        self.get_logger().info("Transition/Memory buffer publisher started")

        # Environment step frequency
        self.declare_parameter("publish_frequency", 10.0)
        self.timer = self.create_timer(
            1.0
            / self.get_parameter("publish_frequency")
            .get_parameter_value()
            .double_value,
            self.timer_callback,
            callback_group=self.cb_group,
        )
        self.add_on_set_parameters_callback(self.parameter_change_callback)

    def parameter_change_callback(self, params: list[rclpy.Parameter]):
        successful = True
        reason = ""

        for param in params:
            if param.name == "publish_frequency":
                # Change timer frequency
                self.timer.cancel()
                self.create_timer(
                    1.0 / param.get_parameter_value().double_value, self.timer_callback
                )
        return SetParametersResult(successful=successful, reason=reason)

    async def timer_callback(self):
        # Send request for action
        s_0 = FloatTensor.build(self.s_0).pack()
        request = srv.PolicyService.Request(s_0=s_0)
        future = self.policy_client.call_async(request)
        try:
            response: srv.PolicyService.Response = await future
            a = FloatTensor.unpack(response.a)
        except Exception as e:
            self.get_logger().warn(str(e))
        # Step environment
        a = a.numpy()
        if a.shape == ():
            a = int(a)
        s_1, r, d, t, _ = self.env.step(a)
        # Pack and send
        msg = Transition.build((self.s_0, a, r, s_1, d))
        self.publisher.publish(msg.pack())
        # Reset env if needed
        if d or t:
            self.s_0, _ = self.env.reset()
        else:
            self.s_0 = s_1


def main(args=None):
    rclpy.init(args=args)
    node = GymController("CartPole-v1")
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
