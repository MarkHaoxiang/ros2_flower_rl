import rclpy
from rclpy.node import Node

from datasets import Dataset
import torch

import ml_interfaces.msg as msg
from ml_interfaces_py import FloatTensor

class DatasetPublisher(Node):
    """ Publishes a Huggingface dataset on a topic
    """

    def __init__(self,
                 name: str,
                 dataset: Dataset):
        super().__init__(name)
        self.dataset = dataset
        self.publisher_ = self.create_publisher(msg.FloatTensor, 'test_tensor_topic', 10)
        self.timer = self.create_timer(0.5, self.timer_callback)

        self.get_logger().info(f"Initialised publisher {name}")
    
    def timer_callback(self):
        msg = FloatTensor.from_torch(torch.rand((2)))
        self.publisher_.publish(msg.pack()) 

def main(args=None):
    rclpy.init(args=args)

    dataset_publisher = DatasetPublisher("test", None)
    rclpy.spin(dataset_publisher)
    dataset_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
