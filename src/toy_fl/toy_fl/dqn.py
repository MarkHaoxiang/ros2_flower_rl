# ruff: noqa: F401

from typing import Any, Callable, Tuple, Type

import flwr as fl
import kitten
import numpy as np
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
from kitten.policy import Policy
from kitten.rl import Algorithm
from kitten.rl.dqn import DQN
from ml_interfaces import msg, srv
from ml_interfaces_py import FloatTensor, RosKnowledge

from rl_actor import RlActor
from ros_kitten_client import RosKittenClient

shard_names = ["critic", "critic_target"]


def build_algorithm(
    cfg: Config,
    knowledge: DQNKnowledge,
    action_space: Space,
    seed: int | None,
    device: str = "cpu",
) -> Tuple[Algorithm, Policy]:
    seed = seed if seed else 0
    rng = kitten.common.global_seed(seed).numpy
    cfg.get("algorithm", {}).pop("critic", None)  # type: ignore
    algorithm = DQN(
        critic=knowledge.critic.net,  # type: ignore
        device=device,
        **cfg.get("algorithm", {}),  # type: ignore
    )
    policy = kitten.policy.EpsilonGreedyPolicy(
        fn=algorithm.policy_fn,
        action_space=action_space,
        rng=rng,
        device=device,
    )
    # Synchronisation
    algorithm._critic = knowledge.critic  # type: ignore
    return algorithm, policy


class DqnActor(RlActor[np.ndarray, np.ndarray]):
    def __init__(
        self,
        node_name: str,
        policy_service: str,
        policy_update_topic: str,
        config: Config,
        knowledge: DQNKnowledge,
        action_space: Space,
        algorithm_factory_fn: Callable,
        seed: int | None = None,
        device: str = "cpu",
    ):
        super().__init__(node_name, policy_service, policy_update_topic)
        self._knowl = knowledge
        self._policy_factory = lambda knowledge: algorithm_factory_fn(
            config, knowledge, action_space, seed, device
        )
        self._algorithm, self._policy = self._policy_factory(self._knowl)

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

    def update_policy(self, msg: msg.Knowledge) -> None:
        shards = RosKnowledge.unpack_to_shards(msg)
        self._knowl.update_knowledge(shards, shard_filter=shard_names)

        # FIXME: I'm not sure if kitten.rl.Algorithm.policy_fn is stateful i.e.
        # if I've already update the critic embeded in an algorithm, do I need
        # to explicitly regenerate the policy and algorithm?
        # If not, remove the line below
        self._algorithm, self._policy = self._policy_factory(self.knowledge)


class DQNRosClient(RosKittenClient):
    def __init__(
        self,
        node_name: str,
        replay_buffer_service: str,
        policy_update_topic: str,
        knowledge: DQNKnowledge,
        action_space: Space[Any],
        config: Config,
        algorithm_factory_fn: Callable,
        seed: int | None = None,
        enable_evaluation: bool = False,
        device: str = "cpu",
    ):
        super().__init__(
            node_name,
            replay_buffer_service,
            policy_update_topic,
            knowledge,
            action_space,
            config,
            seed,
            enable_evaluation,
            device,
        )
        self._policy_factory = lambda knowledge: algorithm_factory_fn(
            config, knowledge, action_space, seed, device
        )

    def _build_algorithm(self) -> None:
        self._algorithm, self._policy = self._policy_factory(self._knowl)

    @property
    def policy(self) -> kitten.policy.Policy:
        return self._policy

    @property
    def algorithm(self) -> kitten.rl.Algorithm:
        return self._algorithm
