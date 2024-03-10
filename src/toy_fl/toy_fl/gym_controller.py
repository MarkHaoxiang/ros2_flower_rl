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

# TODO: put this in a centralized place to retain consistency
# between client and server
ActionType = np.ndarray
StateType = np.ndarray
srv_name = "gym_environment"


class GymController(Node):
    """A robot controller simulating an Gymnasium RL Environment"""

    def __init__(self):
        super().__init__("Gymnasium simulation server")
        self.get_logger().info(f"Building server {self.get_fully_qualified_name()}")
        self.declare_parameters(
            namespace="", parameters=[("env_name", rclpy.Parameter.Type.STRING)]
        )

        name = self.gym_env_name
        env = build_env(name)
        self._env = env
        self.service = self.create_service(
            srv.ControllerService,
            srv_name=srv_name,
            callback=self.on_action_callback,
        )

    def on_action_callback(
        self,
        request: srv.ControllerService.Request,
        response: srv.ControllerService.Response,
    ) -> srv.ControllerService.Response:
        """Callback function to run when we receive an action request

        (IMPORTANT, TODO) this function also has the side effect of
        publishing the transition to the memory/replay buffer node

        Args:
            request (srv.ControllerService.Request): request message received
            response (srv.ControllerService.Response): response message to send

        --------
        Returns:
            srv.ControllerService.Response response message containing the new
            state in the environment when the requested action is taken; if
            the environment is truncated or terminated, the state after reset
            is sent
        """
        action = ControllerService.unpack_request(request, type=ActionType)
        s_1 = self._env_step(action)
        ControllerService.set_response(response, s_1=s_1)
        return response

    def _env_step(self, action: ActionType) -> StateType:
        """Take one step on the included gymnasium environment

        (IMPORTANT, TODO) this function also has the side effect of
        publishing the transition to the memory/replay buffer node

        Args:
            action (ActionType): the action requested by the service


        --------
        Returns:
            The state after the transition is taken; if the environment
            needs to reset after the action is taken, the state after reset
            is returned
        """
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
