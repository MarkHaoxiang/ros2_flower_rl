# ruff: noqa: F401

from typing import Any, Callable, Tuple, Type, Optional, Union

from florl.common import Knowledge
import flwr as fl
from flwr.common import FitIns, FitRes, GetParametersIns, GetParametersRes
import kitten
import numpy as np
from florl.client.kitten.dqn import DQNKnowledge
from flwr.common.typing import Config
from gymnasium.spaces import Space
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup

from toy_frl.frl_client import RosKittenClient

import toy_frl.dqn.common as common

from flwr.client.app import _check_actionable_client, _init_connection
from flwr.client.client import Client
from flwr.client.client_app import ClientApp
from flwr.client.message_handler.message_handler import handle_control_message
from flwr.client.node_state import NodeState
from flwr.client.typing import ClientFn
from flwr.common import GRPC_MAX_MESSAGE_LENGTH
from flwr.common.retry_invoker import RetryInvoker, exponential

class DQNRosClient(RosKittenClient):
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
        self._cb_group = ReentrantCallbackGroup()
        self._init_client()

    def _init_client(self):
        self.get_logger().info("Ros Client started; starting flower client")
        self.start_client(server_address=common.SERVER_ADDR,
                          client=RosClientWrapper(self), insecure=True)


    def start_client(
        self,
        server_address: str,
        client_fn: Optional[ClientFn] = None,
        client: Optional[Client] = None,
        grpc_max_message_length: int = GRPC_MAX_MESSAGE_LENGTH,
        root_certificates: Optional[Union[bytes, str]] = None,
        insecure: Optional[bool] = None,
        transport: Optional[str] = None,
        max_retries: Optional[int] = None,
        max_wait_time: Optional[float] = None,
    ):
        if insecure is None:
            insecure = root_certificates is None

        _check_actionable_client(client, client_fn)
        if client_fn is None:
            assert client is not None
            client_fn = lambda cid: client
        load_client_app_fn = lambda: ClientApp(client_fn=client_fn)
        self.connection, address, connection_error_type = _init_connection(
            transport, server_address
        )

        retry_invoker = RetryInvoker(
            wait_factory=exponential,
            recoverable_exceptions=connection_error_type,
            max_tries=max_retries,
            max_time=max_wait_time,
            # TODO: add handlers for success, failure and stetback
        )

        self.node_state = NodeState()
        self._training_ended = False
        self._train_timer = None
        self.client_manager_timer = self.create_timer(
            timer_period_sec=1.0,
            callback=lambda: self.timed_connect_callback(
                address=address,
                insecure=insecure,
                root_certificates=root_certificates,
                retry_invoker=retry_invoker,
                grpc_max_message_length=grpc_max_message_length,
                load_client_app_fn=load_client_app_fn,
            ),
            callback_group=self._cb_group,
        )
        self.get_logger().info("Client Manager Timer started")

    def timed_connect_callback(
        self,
        address: str,
        insecure: bool,
        root_certificates,
        retry_invoker,
        grpc_max_message_length,
        load_client_app_fn: Callable[[], ClientApp],
    ):
        self.get_logger().info("Client Manager callback called")

        def timer_running(timer):
            return timer is not None and not timer.is_canceled()

        if not timer_running(self._train_timer) and not self._training_ended:
            self.conn = self.connection(address, insecure, retry_invoker, grpc_max_message_length, root_certificates)
            receive, send, create_node, delete_node = self.conn.__enter__()
            # Register node
            if create_node is not None:
                create_node()  # pylint: disable=not-callable
            if self._train_timer is not None:
                self._train_timer.reset()
            else:
                self._train_timer = self.create_timer(
                    timer_period_sec=3.0,
                    callback=lambda: self.timed_message_callback(
                        receive=receive, send=send, load_client_app_fn=load_client_app_fn
                    ),
                    callback_group=self._cb_group,
                )
        elif timer_running(self._train_timer) and self._training_ended:
            self.conn.__exit__(None, None, None)
            assert (
                self._train_timer is not None
            )  # Unneccesary, trying to get the linter happy
            self._train_timer.cancel()
            self.get_logger().info("Training stopped by control message")
        elif timer_running(self._train_timer) and not self._training_ended:
            self.get_logger().info("Flower clients are active")
        else:
            self.get_logger().info("Training has been suspended or terminated")

    def timed_message_callback(self, receive, send, load_client_app_fn):
        self.get_logger().info("Timed message callback entered")
        message = receive()
        self.get_logger().info("messaged received")
        if message is None:
            return
        self.get_logger().info(
            f"[RUN { message.metadata.run_id }, ROUND { message.metadata.group_id }]"
        )
        self.get_logger().info(
            f"Received: { message.metadata.message_type } message { message.metadata.message_id }"
        )
        # Handle control message
        out_message, _ = handle_control_message(message)
        if out_message:
            send(out_message)
            self._training_ended = True
            return
        # Register context for this run
        self.node_state.register_context(run_id=message.metadata.run_id)
        # Retrieve context for this run
        context = self.node_state.retrieve_context(run_id=message.metadata.run_id)
        client_app: ClientApp = load_client_app_fn()
        out_message = client_app(message=message, context=context)
        self.get_logger().info("Out message is ready")
        # Update node state
        self.node_state.update_context(
            run_id=message.metadata.run_id,
            context=context,
        )

        # Send
        send(out_message)
        self.get_logger().info(
            f"[RUN {out_message.metadata.run_id}, ROUND {out_message.metadata.group_id}]"
        )
        self.get_logger().info(
            f"Sent: {out_message.metadata.message_type} reply to message {message.metadata.message_id}"
        )

    def get_parameters(self, ins):
        return self.get_flwr_parameter(ins)

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
        return self._client.get_flwr_parameter(ins)

    def fit(self, ins: FitIns) -> FitRes:
        return self._client.fit(ins)


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
