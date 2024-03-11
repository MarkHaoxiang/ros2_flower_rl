# ruff: noqa: F401
from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Dict, Tuple, Any, List

import flwr as fl
from flwr.common.typing import (GetParametersRes, GetParametersIns)
import kitten
from kitten.experience import AuxiliaryMemoryData
import numpy as np
import torch
from florl.client.client import FlorlClient
from florl.common import Knowledge
from flwr.common.typing import (Config, Scalar)
from gymnasium.spaces import Space

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
        self.take_step(n=1, init=True) # Take first step, initialises the environment

        self._step_cnt = 0
        self._build_algorithm()

        self.early_start()

    def train(self, config: Config) -> Tuple[int, Dict[str, Scalar]]:
        metrics = {}
        # Synchronise critic net
        critic_loss = []
        # Training
        for _ in range(int(config["frames"])):
            self.take_step(n=1)
            # Collected Transitions
            num_samples = int(self._cfg["train"]["minibatch_size"])  # type: ignore
            batch = self.sample_request(num_samples)

            # Placeholder values that doesn't do anything
            place_holder_idxs = torch.zeros(size=(num_samples,), device=self._device)
            placeholder_aux = AuxiliaryMemoryData(
                weights=torch.ones(num_samples, device=self._device),
                random=torch.ones(num_samples, device=self._device),
                indices=place_holder_idxs,
            )
            # End of placeholder values that doesn't do anything

            batch = kitten.experience.Transition(*batch)
            # Algorithm Update
            critic_loss.append(self.algorithm.update(batch, placeholder_aux, self.step))

        # Logging
        metrics["loss"] = sum(critic_loss) / len(critic_loss)
        return 1, metrics

    def take_step(self, n: int, init: bool = False) -> None:
        if init:
            assert n == 1
            action = np.empty((0,), np.float32)
            self._obs = self.action_request(action)
            # NOTE: initialisation does not increment step count, as expected
        else:
            for _ in range(n):
                self._step += 1
                action = self.policy(obs=self._obs)
                self._obs = self.action_request(action)

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

    # TODO: I need to do this
    def early_start(self):
        pass
