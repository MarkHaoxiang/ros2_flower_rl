# ruff: noqa: F401
from __future__ import annotations

from typing import List, Dict, Tuple
from copy import deepcopy
from abc import ABC, abstractmethod

import torch

import kitten
import flwr as fl
from flwr.common.typing import Scalar, Config, GetParametersIns, GetParametersRes

from florl.client.client import FlorlClient
from florl.client.kitten import KittenClient
from florl.common import Knowledge
from rl_actor import RlActor

import numpy as np

action_type = np.ndarray
state_type = np.ndarray

class RosKittenClient(FlorlClient, RlActor[np.ndarray, np.ndarray], ABC):
    def __init__(
        self,
        node_name: str,
        controller_name: str,
        replay_buffer_name: str,
        knowledge: Knowledge,
        config: Config,
        seed: int | None = None,
        enable_evaluation: bool = True,
        device: str = "cpu",
    ):
        # Note currently we cannot support client side evaluation
        FlorlClient.__init__(self, knowledge, enable_evaluation)
        RlActor.__init__(self, node_name, controller_name, replay_buffer_name)
        self._seed = seed if seed else 0 # TODO: What should be the default behaviour here?
        self._cfg = deepcopy(config)
        self._cfg = deepcopy(config)
        self._device = device
        self._step = 0

        # Logging
        self._rng = kitten.common.global_seed(self._seed)

        # RL Modules
        self._step_cnt = 0
        self.build_algorithm()

        self.early_start()

    def train(self, config: Config) -> Tuple[int, Dict[str, Scalar]]:
        metrics = {}
        # Synchronise critic net
        critic_loss = []
        # Training
        for _ in range(int(config["frames"])):
            self._step += 1
            # Collected Transitions
            num_samples = int(self._cfg["train"]["minibatch_size"]) # type: ignore
            batch, aux = self._memory.sample(num_samples)
            batch = kitten.experience.Transition(*batch)
            # Algorithm Update
            critic_loss.append(self.algorithm.update(batch, aux, self.step))

        # Logging
        metrics["loss"] = sum(critic_loss) / len(critic_loss)
        return len(self._memory), metrics

    @abstractmethod
    def build_algorithm(self) -> None:
        raise NotImplementedError

    @property
    @abstractmethod
    def algorithm(self) -> kitten.rl.Algorithm:
        raise NotImplementedError

    @property
    @abstractmethod
    def policy(self) -> kitten.policy.Policy:
        raise NotImplementedError

    def early_start(self):
        pass
