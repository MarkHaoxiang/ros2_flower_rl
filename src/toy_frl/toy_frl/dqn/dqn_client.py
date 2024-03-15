# ruff: noqa: F401

from typing import Any, Callable, Tuple, Type

from florl.common import Knowledge
import flwr as fl
from flwr.common import FitIns, FitRes, GetParametersIns, GetParametersRes
import kitten
import numpy as np
from florl.client.kitten.dqn import DQNKnowledge
from flwr.common.typing import Config
from gymnasium.spaces import Space
import rclpy

from ..frl_client import RosKittenClient

import common

class DQNRosClient(RosKittenClient):
    def __init__(
        self,
        node_name: str,
        replay_buffer_service: str,
        policy_update_topic: str,
        algorithm: kitten.rl.Algorithm,
        policy: kitten.policy.Policy,
        knowledge: Knowledge,
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
            device,
        )

    def _init_client(self):
        self.get_logger().info("Ros Client started; starting flower client")
        fl.client.start_client(server_address=common.SERVER_ADDR,
                               client=RosClientWrapper(self), insecure=True)


    @property
    def policy(self) -> kitten.policy.Policy:
        return self._policy

    @property
    def algorithm(self) -> kitten.rl.Algorithm:
        return self._algorithm

class RosClientWrapper(fl.client.Client):
    def __init__(self, client: DQNRosClient) -> None:
        super().__init__()
        self._client = client

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        return self._client.get_parameters(ins)

    def fit(self, ins: FitIns) -> FitRes:
        return self._client.fit(ins)


def main(args=None):
    rclpy.init(args=args)
    algorithm, policy = common.build_dqn_algorithm(cfg=common.config, knowledge=common.default_knowledge_fn(), action_space=common.action_space, seed=common.SEED, device="cpu")
    node = DQNRosClient(node_name="dqn_actor",
                        replay_buffer_service=common.MEMORY_SERVICE,
                        policy_update_topic=common.POLICY_UPDATE_TOPIC,
                        algorithm=algorithm,
                        policy=policy,
                        knowledge=common.default_knowledge_fn())
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
