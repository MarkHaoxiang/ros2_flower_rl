import math
from collections import OrderedDict
from typing import List

import flwr as fl
import rclpy
import torch
from flwr.common.logger import log
from flwr.common.typing import (Code, FitIns, FitRes, GetParametersIns,
                                GetParametersRes, Parameters, Status)
from rclpy.node import Node

# TODO: Configuration of training parameters
BATCH_SIZE = 16

from typing import Callable, ContextManager, Optional, Tuple, Type, Union

import ml_interfaces.msg as msg
from flwr.client.app import _check_actionable_client, _init_connection
from flwr.client.client import Client
from flwr.client.client_app import ClientApp
from flwr.client.message_handler.message_handler import handle_control_message
from flwr.client.node_state import NodeState
from flwr.client.typing import ClientFn
from flwr.common import GRPC_MAX_MESSAGE_LENGTH
from flwr.common.retry_invoker import RetryInvoker, exponential
from ml_interfaces_py import FeatureLabelPair

from .nn import MnistClassifier


class ToyClient(Node):
    """ROS node responsible for subscribing to sensor data, training and communicating with the flower server"""

    def __init__(self, server_addr: str = "[::]:8080"):
        """ROS node responsible for subscribing to sensor data, training and communicating with the flower server

        Args:
            server_addr (str, optional): Flower server sonnection address. Defaults to "[::]:8080".
        """
        super().__init__("toy_client")
        self.get_logger().info(f"Building client {self.get_fully_qualified_name()}")

        self._subscriber = self.create_subscription(
            msg_type=msg.FeatureLabelPair,
            topic="data_stream",
            callback=self.listener_callback,
            qos_profile=10,
        )
        self._server_addr = server_addr

        # Data
        self._feature_buffer = []
        self._label_buffer = []
        self._X = None
        self._y = None
        # Net
        self._net = MnistClassifier()
        self._loss_fn = torch.nn.CrossEntropyLoss()
        self._optim = torch.optim.Adam(self._net.parameters())

    def listener_callback(self, msg: msg.FeatureLabelPair) -> None:
        """Callback to buffer datastream

        Args:
            msg (FeatureLabelPair): Subscribed data recieved.
        """
        msg = FeatureLabelPair.unpack(msg)
        feature, label = msg.torch()
        self._feature_buffer.append(feature)
        self._label_buffer.append(label)

        # Check for initialisation
        # TODO: Change init time
        if self._X is None and len(self._feature_buffer) >= BATCH_SIZE * 10:
            self.initialise_client()

    def initialise_client(self):
        self.get_logger().info("Data ready. Initialising Flower Client.")
        self._X = torch.stack(self._feature_buffer)
        self._y = torch.stack(self._label_buffer)
        self.start_client(
            server_address=self._server_addr,
            client=ToyClientWrapper(self),
            insecure=True,
        )

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

        message = receive()
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
        # Load ClientApp instance
        client_app: ClientApp = load_client_app_fn()
        # Handle task message
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

    def flwr_get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        return GetParametersRes(
            status=Status(code=Code.OK, message=""),
            parameters=fl.common.ndarrays_to_parameters(
                [val.cpu().numpy() for _, val in self._net.state_dict().items()]
            ),
        )

    def flwr_set_parameters(self, parameters: Parameters) -> None:
        nd_arrays = fl.common.parameters_to_ndarrays(parameters)
        params_dict = zip(self._net.state_dict().keys(), nd_arrays)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self._net.load_state_dict(state_dict, strict=True)

    def fit(self, ins: FitIns) -> FitRes:
        # Collect data
        self.get_logger().info("Fit is called")
        if len(self._feature_buffer) > 0:
            features = torch.stack(self._feature_buffer)
            labels = torch.stack(self._label_buffer)
            self._X = torch.cat((self._X, features))
            self._y = torch.cat((self._y, labels))
            self._feature_buffer = []
            self._label_buffer = []

        self.get_logger().info("Fit is called; bp1")
        # Update parameters
        self.flwr_set_parameters(ins.parameters)

        # Train
        n = len(self._X)
        n_batches = math.ceil(n / BATCH_SIZE)
        average_loss = 0
        self.get_logger().info("Fit is called; bp2")
        for i in range(n_batches):
            self.get_logger().info(f"Fit is called; bp3, iteration {i}")
            X = self._X[i * BATCH_SIZE : max(n, i + BATCH_SIZE)]
            y = self._y[i * BATCH_SIZE : max(n, i + BATCH_SIZE)].to(torch.long)
            py = self._net(X)
            loss = self._loss_fn(py, y)
            average_loss += loss.item()
            self._optim.zero_grad()
            loss.backward()
            self._optim.step()
        average_loss = average_loss / n_batches

        self.get_logger().info("Fit is called; bp4")
        # Results
        return FitRes(
            status=Status(Code.OK, message=""),
            num_examples=n,
            parameters=self.flwr_get_parameters(
                GetParametersIns(config=ins.config)
            ).parameters,
            metrics={"loss": average_loss},
        )

    @property
    def device(self) -> str:
        """Training device"""
        return "cpu"


class ToyClientWrapper(fl.client.Client):
    """Compatibility layer between ROS and Flower"""

    def __init__(self, toy_client: ToyClient) -> None:
        """Provides a correct namespace mapping for toy_client due to collisions between Ros and Flower

        Args:
            toy_client (ToyClient): _description_
        """
        super().__init__()
        self._client = toy_client

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        return self._client.flwr_get_parameters(ins)

    def fit(self, ins: FitIns) -> FitRes:
        return self._client.fit(ins)


def main(args=None):
    rclpy.init(args=args)
    node = ToyClient()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
