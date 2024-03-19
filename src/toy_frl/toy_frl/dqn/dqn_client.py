# ruff: noqa: F401
from florl.common import Knowledge
import flwr as fl
from flwr.common import FitIns, FitRes, GetParametersIns, GetParametersRes
import kitten
import numpy as np
from florl.client.kitten.dqn import DQNKnowledge
from flwr.common.typing import Config
from gymnasium.spaces import Space
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

from toy_frl.dqn import common
from toy_frl.frl_client import FRLClient

class DQNRosClient(FRLClient):
    def __init__(
        self,
        node_name: str,
        replay_buffer_service: str,
        policy_update_topic: str,
        algorithm: kitten.rl.Algorithm,
        policy: kitten.policy.Policy,
        knowledge: Knowledge,
        config,
        # action_space: Space[Any],
        device: str = "cpu",
    ):
        self._policy = policy
        super().__init__(
            node_name,
            replay_buffer_service,
            policy_update_topic,
            algorithm,
            knowledge,
            config,
            device,
        )
        self._cb_group = MutuallyExclusiveCallbackGroup()
        self.start_client()

    @property
    def policy(self) -> kitten.policy.Policy:
        return self._policy

    @property
    def algorithm(self) -> kitten.rl.Algorithm:
        return self._algorithm

def main(args=None):
    rclpy.init(args=args)
    knowledge = common.default_knowledge_fn()
    config = common.config["rl"]
    algorithm, policy = common.build_dqn_algorithm(cfg=config, knowledge=knowledge, action_space=common.action_space, seed=common.SEED, device="cpu")
    node = DQNRosClient(node_name="dqn_actor",
                        replay_buffer_service=common.MEMORY_SERVICE,
                        policy_update_topic=common.POLICY_UPDATE_TOPIC,
                        algorithm=algorithm,
                        policy=policy,
                        knowledge=knowledge,
                        config=config)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
