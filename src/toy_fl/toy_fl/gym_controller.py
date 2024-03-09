# ruff: noqa: F401

from typing import List

import rclpy
from rclpy.node import Node

from rcl_interfaces.msg import SetParametersResult

from datasets import load_from_disk

import torch
import numpy as np

import ml_interfaces.msg as msg
import ml_interfaces.srv as srv
from ml_interfaces_py import FeatureLabelPair, ControllerService

from gymnasium.core import Env

import kitten
from kitten.common.util import build_env, build_critic


class GymController(Node):
    """Publishes a HuggingFace dataset on a topic"""

    def __init__(self):
        super().__init__("Gymnasium simulation server")
        self.get_logger().info(f"Building publisher {self.get_fully_qualified_name()}")
        self.declare_parameters(
            namespace="", parameters=[("env_name", rclpy.Parameter.Type.STRING)]
        )

        name = self.gym_env_name
        env = build_env(name)
        self._env = env
        self.services = self.create_service(
            srv.ControllerService,
            srv_name="gym_environment",
            callback=self.on_action_callback,
        )

    def on_action_callback(
        self,
        request: srv.ControllerService.Request,
        response: srv.ControllerService.Response,
    ) -> srv.ControllerService.Response:
        action = ControllerService.unpack_request(request, type=np.ndarray)
        s_1 = self.env_step(action)
        ControllerService.set_response(response, s_1=s_1)
        return response

    def env_step(self, action: np.ndarray):
        # Transition
        s_1, reward, terminated, truncated, info = self._env.step(action)
        # TODO: whenever a step is taken, publish the transition
        if terminated or truncated:
            self._env.reset()
        return s_1

    @property
    def gym_env_name(self) -> str | None:
        return self.get_parameter("env_name").get_parameter_value().string_value


def main(args=None):
    rclpy.init(args=args)
    node = GymController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
