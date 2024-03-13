# ruff: noqa: F401
from __future__ import annotations

import kitten
import numpy as np
import rclpy
import torch
from florl.client.client import FlorlClient
from florl.common import Knowledge
from flwr.common.typing import Config, GetParametersIns, GetParametersRes, Scalar
from ml_interfaces import msg as msg
from ml_interfaces import srv as srv
from ml_interfaces_py import RosKnowledge, Transition
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node

action_type = np.ndarray
state_type = np.ndarray


class RosKittenClient(FlorlClient, Node):
    def __init__(
        self,
        node_name: str,
        replay_buffer_service: str,
        policy_update_topic: str,
        algorithm: kitten.rl.Algorithm,
        knowledge: Knowledge,
        device: str = "cpu",
    ):
        # Note currently we cannot support client side evaluation
        FlorlClient.__init__(self, knowledge, enable_evaluation=False)
        Node.__init__(self, node_name=node_name)
        self._algorithm = algorithm
        self._knowl = knowledge
        self._device = device
        self._step = 0

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
        for _ in range(int(config["frames"])):
            # Collected Transitions
            num_samples = int(self._cfg["train"]["minibatch_size"])  # type: ignore
            # TODO: Return Flower failure if not enough samples
            batch, aux = rclpy.spin_until_future_complete(self.sample_request(num_samples))
            # Algorithm Update
            critic_loss.append(self._algorithm.update(batch, aux, self.step))
        # Sync Policy
        self.publish_knowledge(self._knowl)

        # Logging
        metrics["loss"] = sum(critic_loss) / len(critic_loss)
        return 1, metrics

    def get_flwr_parameter(self, ins: GetParametersIns) -> GetParametersRes:
        return FlorlClient.get_parameters(self, ins)

    def get_ros_parameter(self, names: list[str]):
        return Node.get_parameters(self, names)

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

    def publish_knowledge(self, knowledge: Knowledge) -> None:
        msg = RosKnowledge.pack(knowledge)
        self.policy_publisher.publish(msg)
