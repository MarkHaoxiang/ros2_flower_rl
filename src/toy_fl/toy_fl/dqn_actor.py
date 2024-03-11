# ruff: noqa: F401

from typing import Any, Type

import flwr as fl
import kitten
from florl.client.kitten.dqn import DQNKnowledge
from flwr.common.typing import (
    Config,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    Scalar,
)
from gymnasium.spaces import Space
from kitten.common.util import build_env
from kitten.rl.dqn import DQN

from ros_kitten_client import RosKittenClient


class DqnActor(RosKittenClient):
    def __init__(self, *args, **kwargs):
        assert (knowledge := kwargs.get("knowledge", None)) is not None and isinstance(
            knowledge, DQNKnowledge
        )

        super().__init__(*args, **kwargs)

    def _build_algorithm(self) -> None:
        self._cfg.get("algorithm", {}).pop("critic", None)  # type: ignore
        self._algorithm = DQN(
            critic=self.knowledge.critic.net,  # type: ignore
            device=self._device,
            **self._cfg.get("algorithm", {}),  # type: ignore
        )
        self._policy = kitten.policy.EpsilonGreedyPolicy(
            fn=self.algorithm.policy_fn,
            action_space=self._action_space,
            rng=self._rng.numpy,
            device=self._device,
        )
        # Synchronisation
        self._algorithm._critic = self.knowledge.critic  # type: ignore

    @property
    def algorithm(self) -> kitten.rl.Algorithm:
        return self._algorithm

    @property
    def policy(self) -> kitten.policy.Policy:
        return self._policy

    @property
    def knowledge(self) -> DQNKnowledge:
        assert isinstance(self._knowl, DQNKnowledge)
        return self._knowl


class DqnActorWrapper(fl.client.Client):
    def __init__(self, dqn_actor: DqnActor):
        super().__init__()
        self._actor = dqn_actor

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        return self._actor.get_flwr_parameter(ins)

    def fit(self, ins: FitIns) -> FitRes:
        return self._actor.fit(ins)
