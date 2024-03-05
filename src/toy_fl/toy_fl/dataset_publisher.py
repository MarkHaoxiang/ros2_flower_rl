from typing import List

import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import SetParametersResult

from datasets import load_from_disk
import torch

import ml_interfaces.msg as msg
from ml_interfaces_py import FeatureLabelPair


class DatasetPublisher(Node):
    """Publishes a HuggingFace dataset on a topic"""

    def __init__(self):
        super().__init__("dataset_publisher")
        self.get_logger().info(f"Building publisher {self.get_fully_qualified_name()}")

        # Declare Parameters
        self.declare_parameters(
            namespace="",
            parameters=[
                ("dataset_dir", rclpy.Parameter.Type.STRING),
                ("publish_frequency", 10.0),
                ("feature_identifier", "image"),
                ("label_identifier", "label"),
            ],
        )
        self.timer = self.create_timer(
            1.0
            / self.get_parameter("publish_frequency")
            .get_parameter_value()
            .double_value,
            self.timer_callback,
        )
        self._dataset_dir = self.dataset_dir_parameter
        self._dataset = None
        self._dataset_index = 0

        self.add_on_set_parameters_callback(self.parameter_change_callback)

        # Build publisher
        self._publisher = self.create_publisher(msg.FeatureLabelPair, "data_stream", 10)

    def parameter_change_callback(self, params: List[rclpy.Parameter]):
        successful = True
        reason = ""

        for param in params:
            if param.name == "publish_frequency":
                # Change timer frequency
                self.timer.cancel()
                self.create_timer(
                    1.0 / param.get_parameter_value().double_value, self.timer_callback
                )

        return SetParametersResult(successful=successful, reason=reason)

    def timer_callback(self):
        # Load dataset
        if not self.dataset_dir_parameter is None:
            if self._dataset is None or self.dataset_dir_parameter != self._dataset_dir:
                self._dataset_dir = self.dataset_dir_parameter
                self.get_logger().info(f"Loading dataset {self._dataset_dir}")
                self._dataset = load_from_disk(self._dataset_dir).with_format("torch")
        else:
            self._dataset = None

        # No dataset provided
        if self._dataset is None:
            return

        # Publish dataset
        self._dataset_index = (self._dataset_index + 1) % len(self._dataset)
        feature_identifier = (
            self.get_parameter("feature_identifier").get_parameter_value().string_value
        )
        label_identifier = (
            self.get_parameter("label_identifier").get_parameter_value().string_value
        )

        data = self._dataset[self._dataset_index]
        X: torch.Tensor = data[feature_identifier].to(torch.float32)
        y: torch.Tensor = data[label_identifier].to(torch.float32)
        msg = FeatureLabelPair.build(X, y)
        self._publisher.publish(msg.pack())

    @property
    def dataset_dir_parameter(self) -> str | None:
        return self.get_parameter("dataset_dir").get_parameter_value().string_value


def main(args=None):
    rclpy.init(args=args)
    node = DatasetPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
