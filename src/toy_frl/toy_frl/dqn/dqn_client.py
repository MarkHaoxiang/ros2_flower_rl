# ruff: noqa: F401
from florl.common import Knowledge
from flwr.common import FitIns, FitRes
import kitten
from flwr.common.typing import Config
import rclpy
from rclpy import executors
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
        config: Config,
        server_addr: str = "[::]:8080",
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
            server_addr,
            device,
        )
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
    executor = executors.MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.destroy_node()
        executor.shutdown()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
