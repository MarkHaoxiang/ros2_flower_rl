import logging
import math
from collections import OrderedDict
from typing import List

import flwr as fl
import rclpy
import torch
from flwr.common.logger import log
from flwr.common.typing import (
    Code,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    Parameters,
    Status,
)
from rclpy.node import Node

# TODO: Configuration of training parameters
BATCH_SIZE = 16

import ml_interfaces.msg as msg
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
        fl.client.start_client(
            server_address=self._server_addr,
            client=ToyClientWrapper(self),
            insecure=True,
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
        if len(self._feature_buffer) > 0:
            features = torch.stack(self._feature_buffer)
            labels = torch.stack(self._label_buffer)
            self._X = torch.cat((self._X, features))
            self._y = torch.cat((self._y, labels))
            self._feature_buffer = []
            self._label_buffer = []

        # Update parameters
        self.flwr_set_parameters(ins.parameters)

        # Train
        n = len(self._X)
        n_batches = math.ceil(n / BATCH_SIZE)
        average_loss = 0
        for i in range(n_batches):
            X = self._X[i * BATCH_SIZE : max(n, i + BATCH_SIZE)]
            y = self._y[i * BATCH_SIZE : max(n, i + BATCH_SIZE)].to(torch.long)
            py = self._net(X)
            loss = self._loss_fn(py, y)
            average_loss += loss.item()
            self._optim.zero_grad()
            loss.backward()
            self._optim.step()
        average_loss = average_loss / n_batches

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
