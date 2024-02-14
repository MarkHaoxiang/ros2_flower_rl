import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from datasets import Dataset

class DatasetPublisher(Node):
    """ Publishes a Huggingface dataset on a topic
    """
    def __init__(self,
                 name: str,
                 dataset: Dataset):
        super().__init__(name)
        self.dataset = dataset
        self.publisher_ = self.create_publisher(String, 'topic', 10)
        self.timer = self.create_timer(0.5, self.timer_callback)
    
    def timer_callback(self):
        msg = String()
        msg.data = "test"
        self.publisher_.publish(msg) 

def main(args=None):
    rclpy.init(args=args)

    dataset_publisher = DatasetPublisher("test", None)
    rclpy.spin(dataset_publisher)
    dataset_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
