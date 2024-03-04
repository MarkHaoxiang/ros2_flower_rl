from rclpy.node import Node

from datasets import Dataset
import torch

import ml_interfaces.msg as msg
from ml_interfaces_py import FeatureLabelPair

class DatasetPublisher(Node):
    """ Publishes a Huggingface dataset on a topic
    """

    def __init__(self,
                 name: str,
                 dataset: Dataset,
                 frequency: float = 2.0,
                 feature_identifier: str = "img",
                 label_identifier: str = "label"):
        super().__init__(name)
        self.dataset = dataset.with_format("torch")
        self.dataset_index = 0
        self.publisher_ = self.create_publisher(msg.FeatureLabelPair, 'test_tensor_topic', 10)
        self.timer = self.create_timer(1 / frequency, self.timer_callback)
        self.feature_identifier = feature_identifier
        self.label_identifier = label_identifier

        self.get_logger().info(f"Initialised publisher {name}")
    
    def timer_callback(self):
        # Update dataset index
        self.dataset_index = (self.dataset_index + 1) % len(self.dataset)
        data = self.dataset[self.dataset_index]
        X: torch.Tensor = data[self.feature_identifier].to(torch.float32)
        y: torch.Tensor = data[self.label_identifier].to(torch.float32)
        msg = FeatureLabelPair.build(X, y)
        self.publisher_.publish(msg.pack()) 
