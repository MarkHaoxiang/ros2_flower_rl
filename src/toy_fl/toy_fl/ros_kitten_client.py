# ruff: noqa: F401
from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Dict, List, Tuple

import kitten
import numpy as np
import rclpy
import torch
from florl.client.client import FlorlClient
from florl.common import Knowledge
from flwr.common.typing import Config, GetParametersIns, GetParametersRes, Scalar
from gymnasium.spaces import Space
from kitten.experience import AuxiliaryMemoryData
from ml_interfaces import msg as msg
from ml_interfaces import srv as srv
from ml_interfaces_py import RosKnowledge, Transition
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node

from rl_actor import RlActor

action_type = np.ndarray
state_type = np.ndarray


class RosKittenClient(FlorlClient, Node, ABC):
    def __init__(
        self,
        node_name: str,
        replay_buffer_service: str,
        policy_update_topic: str,
        knowledge: Knowledge,
        action_space: Space[Any],
        config: Config,
        seed: int | None = None,
        enable_evaluation: bool = False,
        device: str = "cpu",
    ):
        # Note currently we cannot support client side evaluation
        FlorlClient.__init__(self, knowledge, enable_evaluation)
        Node.__init__(self, node_name=node_name)  # type: ignore
        self._knowl = knowledge
        self._seed = (
            seed if seed else 0
        )  # TODO: What should be the default behaviour here?
        self._cfg = deepcopy(config)
        self._device = device
        self._step = 0
        self._action_space = action_space

        # Logging
        self._rng = kitten.common.global_seed(self._seed)

        self._build_algorithm()

        self._cb_group = ReentrantCallbackGroup()
        self.memory_client = self.create_client(
            srv_type=srv.SampleTransition,
            srv_name=replay_buffer_service,
            callback_group=self._cb_group,
        )
        while not self.memory_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Replay Bufferr Service not available, waiting; communication dependent on this: policy_publisher")

        self.policy_publisher = self.create_publisher(
            msg_type=msg.Knowledge,
            topic=policy_update_topic,
            qos_profile=10,
            callback_group=self._cb_group,
        )

    def train(self, config: Config) -> Tuple[int, Dict[str, Scalar]]:
        metrics = {}
        # Synchronise critic net
        critic_loss = []
        # Training
        for _ in range(int(config["frames"])):
            # Collected Transitions
            self._step += 1  # WARNING: Check if this is the intended behaviour
            num_samples = int(self._cfg["train"]["minibatch_size"])  # type: ignore
            # TODO: Deal with async...
            # TODO: Return Flower failure if not enough samples
            batch, aux = self.sample_request(num_samples)
            # Algorithm Update
            critic_loss.append(self.algorithm.update(batch, aux, self.step))

        self.publish_knowledge(self._knowl)

        # Logging
        metrics["loss"] = sum(critic_loss) / len(critic_loss)
        return 1, metrics

    def get_flwr_parameter(self, ins: GetParametersIns) -> GetParametersRes:
        return FlorlClient.get_parameters(self, ins)

    def get_ros_parameter(self, names: List[str]):
        return Node.get_parameters(self, names)

    @property
    def step(self) -> int:
        """Number of collected frames"""
        return self._step

    @property
    def action_space(self) -> Space[Any]:
        """Number of collected frames"""
        return self._action_space

    @abstractmethod
    def _build_algorithm(self) -> None:
        raise NotImplementedError

    @property
    @abstractmethod
    def algorithm(self) -> kitten.rl.Algorithm:
        raise NotImplementedError

    @property
    @abstractmethod
    def policy(self) -> kitten.policy.Policy:
        raise NotImplementedError

    async def sample_request(
        self, n: int
    ) -> tuple[kitten.experience.Transition, kitten.experience.AuxiliaryMemoryData]:
        """Samples a batch from memory

        Args:
            n (int): minibatch size.

        Returns:
            tuple[kitten.experience.Transition, kitten.experience.AuxiliaryMemoryData]: training batch
        """
        request = srv.SampleTransition.Request(n=n)
        future = self.memory_client.call_async(request)
        try:
            response: srv.SampleTransition.Response = await future
        except Exception as e:
            self.get_logger().warn(str(e))

        batch = [Transition.unpack(x) for x in response.batch]
        batch = [x.numpy() for x in batch]
        batch = kitten.experience.util.build_transition_from_list(batch)
        # Placeholder
        aux = kitten.experience.AuxiliaryMemoryData(
            weights=torch.ones(len(batch.s_0), batch.s_0.get_device()),
            random=None,
            indices=None,
        )
        return batch, aux

    def publish_knowledge(self, knowledge: Knowledge) -> None:
        msg = RosKnowledge.pack(knowledge)
        self.policy_publisher.publish(msg)
