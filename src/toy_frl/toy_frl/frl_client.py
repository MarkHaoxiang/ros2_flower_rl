# ruff: noqa: F401
from __future__ import annotations

from flwr.common import EvaluateIns, EvaluateRes, FitIns, FitRes, GetPropertiesIns, GetPropertiesRes
import kitten
import numpy as np
import rclpy
from rclpy.node import Node
import torch
from florl.client.client import FlorlClient
from florl.common import Knowledge
from flwr.common.typing import Config, GetParametersIns, GetParametersRes, Scalar
from ml_interfaces import msg as msg
from ml_interfaces import srv as srv
from ml_interfaces_py import RosKnowledge, Transition
from ros2_flower_bridge import TimerCallbackClient

action_type = np.ndarray
state_type = np.ndarray


class FRLClient(TimerCallbackClient, FlorlClient):
    def __init__(
        self,
        node_name: str,
        replay_buffer_service: str,
        policy_update_topic: str,
        algorithm: kitten.rl.Algorithm,
        knowledge: Knowledge,
        config,
        server_addr: str = "[::]:8080",
        device: str = "cpu",
    ):
        # Note currently we cannot support client side evaluation
        TimerCallbackClient.__init__(self, node_name, server_addr=server_addr)
        FlorlClient.__init__(self, knowledge, enable_evaluation=False)
        self._algorithm = algorithm
        self._knowl = knowledge
        self._device = device
        self._step = 0
        self._cfg = config

        # Memory Service
        self.memory_client = self.create_client(
            srv_type=srv.SampleTransition,
            srv_name=replay_buffer_service,
        )
        while not self.memory_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Replay Buffer Service not available, waiting; communication dependent on this: policy_publisher")

        # Policy Publisher
        self.policy_publisher = self.create_publisher(
            msg_type=msg.Knowledge,
            topic=policy_update_topic,
            qos_profile=10,
        )

    def train(self, config: Config) -> tuple[int, dict[str, Scalar]]:
        metrics = {}
        # Synchronise critic net
        critic_loss = []
        # Training
        self.get_logger().info(f"Begin training round")
        for i in range(int(config["frames"])):
            # Collected Transitions
            num_samples = self._cfg["train"]["minibatch_size"]  # type: ignore
            # TODO: Deal with lower number of samples than requested
            batch, aux = self.sample_request(num_samples)

            # Algorithm Update
            critic_loss.append(self._algorithm.update(batch, aux, self._step))
            self._step += 1
        # Sync Policy
        self.publish_knowledge(self._knowl)
        self.get_logger().info(f"Published updated knowledge")

        # Logging
        metrics["loss"] = sum(critic_loss) / len(critic_loss)
        return 1, metrics

    def sample_request(
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
        rclpy.spin_until_future_complete(self, future)
        response = future.result()
        batch = [Transition.unpack(x) for x in response.batch]
        batch = [x.numpy() for x in batch]
        batch = kitten.experience.util.build_transition_from_list(batch)
        res_batch = kitten.experience.Transition(s_0=batch.s_0, a=batch.a.int(), r=batch.r, s_1=batch.s_1, d=batch.d)
        aux = kitten.experience.AuxiliaryMemoryData(
            weights=torch.ones(len(res_batch.s_0), device=self._device),
            random=None,
            indices=None,
        )
        return res_batch, aux
    
    def fit(self, ins: FitIns) -> FitRes:
        return FlorlClient.fit(self._proxy_client, ins)
    
    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        return FlorlClient.evaluate(self._proxy_client, ins)
    
    def flwr_get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        return FlorlClient.get_parameters(self, ins)
    
    def flwr_get_properties(self, ins: GetPropertiesIns) -> GetPropertiesRes:
        return FlorlClient.get_properties(self, ins)

    def publish_knowledge(self, knowledge: Knowledge) -> None:
        msg = RosKnowledge.pack(knowledge)
        self.policy_publisher.publish(msg)
