from fl_subscriber import FlowerSubscriber
from flwr import Client
from typing import List, Tuple


class RosClient(Client):
    training_data_dir: List[str] = []
    subscriber_node: FlowerSubscriber

    def __init__(self, subscriber_node: FlowerSubscriber):
        super.__init__(self)
        self.subscriber_node = subscriber_node
        self.subscriber_node.set_flower_hook(
            lambda self, training_data: self.training_data_dir.append(training_data)
        )
