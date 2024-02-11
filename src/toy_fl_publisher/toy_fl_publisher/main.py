from abc import ABC

import rclpy
from rclpy.node import Node

class DatasetPublisher(Node, ABC):
    
    def __init__(self, name: str):
        super().__init__(name)

def main(args=None):
    rclpy.init(args=args)

    dataset_publisher = DatasetPublisher()

    rclpy.spin(dataset_publisher)
    dataset_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
