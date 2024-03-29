# ruff: noqa: F401

from typing import Callable

from florl.common import Knowledge
import kitten
import numpy as np
from florl.client.kitten.dqn import DQNKnowledge
from flwr.common.typing import Config
from gymnasium.spaces import Space
from ml_interfaces import msg
from ml_interfaces_py import RosKnowledge
import rclpy

from toy_frl.frl_policy import RlActor

import toy_frl.dqn.common as common

class DqnActor(RlActor[np.ndarray, np.ndarray]):
    def __init__(
        self,
        node_name: str,
        policy_service: str,
        policy_update_topic: str,
        config: Config,
        knowledge: Knowledge,
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
        self.get_logger().info("Policy Update Received")
        shards = RosKnowledge.unpack_to_shards(msg)
        self._knowl.update_knowledge(shards, shard_filter=common.DQN_SHARDS)
        self.get_logger().info("Knowledge updated")

        # FIXME: I'm not sure if kitten.rl.Algorithm.policy_fn is stateful i.e.
        # if I've already update the critic embeded in an algorithm, do I need
        # to explicitly regenerate the policy and algorithm?
        # If not, remove the line below
        self.get_logger().info("Updating Policy")
        self._algorithm, self._policy = self._policy_factory(self.knowledge)
        self.get_logger().info("Policy Updated")

def main(args=None):
    rclpy.init(args=args)
    node = DqnActor(node_name="dqn_actor",
                    policy_service=common.POLICY_SERVICE,
                    policy_update_topic=common.POLICY_UPDATE_TOPIC,
                    config=common.config,
                    knowledge=common.default_knowledge_fn(),
                    action_space=common.action_space,
                    algorithm_factory_fn=common.build_dqn_algorithm,
                    seed=common.SEED)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
