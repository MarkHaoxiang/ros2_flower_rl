import math
import threading
from collections import OrderedDict

import flwr as fl
import rclpy
from rclpy import executors
import torch
from flwr.common.typing import (Code, FitIns, FitRes, GetParametersIns,
                                GetParametersRes, Parameters, Status)
import ml_interfaces.msg as msg
from ml_interfaces_py import FeatureLabelPair
from ros2_flower_bridge import TimerCallbackClient

from .nn import MnistClassifier


BATCH_SIZE = 16
MAX_STORAGE_SIZE = 4096

class ToyClient(TimerCallbackClient):
    """ROS node responsible for subscribing to sensor data, training and communicating with the flower server"""
    def __init__(self, server_addr: str = "[::]:8080"):
        """ROS node responsible for subscribing to sensor data, training and communicating with the flower server

        Args:
            server_addr (str, optional): Flower server sonnection address. Defaults to "[::]:8080".
        """
        super().__init__("toy_client", server_addr=server_addr)
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

        # Safety
        self.lock = threading.Lock()
    
    def listener_callback(self, msg: msg.FeatureLabelPair) -> None:
        """Callback to buffer datastream

        Args:
            msg (FeatureLabelPair): Subscribed data recieved.
        """
        msg = FeatureLabelPair.unpack(msg)
        feature, label = msg.torch()
        with self.lock:
            self._feature_buffer.append(feature)
            self._label_buffer.append(label)

        if self._X is None and len(self._feature_buffer) >= BATCH_SIZE * 10:
            self.initialise_client()

    def initialise_client(self):
        self.get_logger().info("Data ready. Initialising Flower Client.")
        self._X = torch.stack(self._feature_buffer)
        self._y = torch.stack(self._label_buffer)
        self.start_client(insecure=True)

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
        self.get_logger().info(f"Fit with {len(self._X)}")
        # Collect data
        with self.lock:
            if len(self._feature_buffer) > 0:
                features = torch.stack(self._feature_buffer)
                labels = torch.stack(self._label_buffer)
                self._X = torch.cat((self._X, features))
                self._y = torch.cat((self._y, labels))
                self._feature_buffer = []
                self._label_buffer = []
        if len(self._X) > MAX_STORAGE_SIZE:
            self._X = self._X[-MAX_STORAGE_SIZE:]
            self._y = self._y[-MAX_STORAGE_SIZE:]

        # Update parameters
        self.flwr_set_parameters(ins.parameters)

        # Train
        n = len(self._X)
        n_batches = math.ceil(n / BATCH_SIZE)
        average_loss = 0
        for i in range(n_batches):
            self.get_logger().info(f"Fitting batch {i} out of {n_batches} with {len(self._X)}, {len(self._feature_buffer)} in buffer.")
            X = self._X[i * BATCH_SIZE : max(n, i + BATCH_SIZE)]
            y = self._y[i * BATCH_SIZE : max(n, i + BATCH_SIZE)].to(torch.long)
            py = self._net(X)
            loss = self._loss_fn(py, y)
            average_loss += loss.item()
            self._optim.zero_grad()
            loss.backward()
            self._optim.step()
        average_loss = average_loss / n_batches
        return FitRes(
            status=Status(Code.OK, message=""),
            num_examples=n,
            parameters=self.flwr_get_parameters(
                GetParametersIns(config=ins.config)
            ).parameters,
            metrics={"loss": average_loss},
        ) 

def main(args=None):
    rclpy.init(args=args)
    node = ToyClient()
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
