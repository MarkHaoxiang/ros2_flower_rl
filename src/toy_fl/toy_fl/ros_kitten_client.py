# ruff: noqa: F401
from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Dict, List, Tuple

import rclpy
import kitten
import numpy as np
import torch
from florl.client.client import FlorlClient
from florl.common import Knowledge
from flwr.common.typing import Config, GetParametersIns, GetParametersRes, Scalar
from gymnasium.spaces import Space
from ml_interfaces import srv as srv
from ml_interfaces import msg as msg
from ml_interfaces_py import Transition
from kitten.experience import AuxiliaryMemoryData

from rl_actor import RlActor

action_type = np.ndarray
state_type = np.ndarray


class RosKittenClient(FlorlClient, RlActor[np.ndarray, np.ndarray], ABC):
    def __init__(
        self,
        node_name: str,
        controller_name: str,
        replay_buffer_name: str,
        knowledge: Knowledge,
        action_space: Space[Any],
        config: Config,
        seed: int | None = None,
        enable_evaluation: bool = False,
        device: str = "cpu",
    ):
        # Note currently we cannot support client side evaluation
        FlorlClient.__init__(self, knowledge, enable_evaluation)
        RlActor.__init__(self, node_name, controller_name, replay_buffer_name)
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

        # RL Modules
        # For initialisation
        self._obs = np.empty(0, np.float32)
        self.take_step(n=1, init=True)  # Take first step, initialises the environment

        self._step_cnt = 0
        self._build_algorithm()

        self.memory_client = self.create_client(
            srv_type=srv.SampleTransition, srv_name=replay_buffer_name
        )

    def train(self, config: Config) -> Tuple[int, Dict[str, Scalar]]:
        metrics = {}
        # Synchronise critic net
        critic_loss = []
        # Training
        for _ in range(int(config["frames"])):
            # Collected Transitions
            num_samples = int(self._cfg["train"]["minibatch_size"])  # type: ignore
            # TODO: Deal with async...
            # TODO: Return Flower failure if not enough samples
            batch, aux = self.sample_request(num_samples)
            # Algorithm Update
            critic_loss.append(self.algorithm.update(batch, aux, self.step))

        # Logging
        metrics["loss"] = sum(critic_loss) / len(critic_loss)
        return 1, metrics

    def get_flwr_parameter(self, ins: GetParametersIns) -> GetParametersRes:
        return FlorlClient.get_parameters(self, ins)

    def get_ros_parameter(self, names: List[str]):
        return RlActor[np.ndarray, np.ndarray].get_parameters(self, names)

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

