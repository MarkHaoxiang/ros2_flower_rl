# import rclpy
from rclpy.node import Node, Subscription
from ml_interfaces_py.lib import FeatureLabelPair
from typing import List, Tuple, Optional

import torch
from torch import Tensor
from os import path
from datatime import datetime

from flwr_client import RosClient
import flwr as fl

# Simple id counter to assign unique id for each node


def _raise(ex: Exception):
    """Wrapper function to raise exceptions in lambda"""
    raise ex


class FlowerSubscriber(Node):
    """
    ROS node responsible for feeding data sets to flower client,
    communicates with the flower client via a hook function
    """

    count: int = 0  # Global ID count-up

    def __init__(
        self,
        flower_hook_fn: Optional[RosClient.flower_hook_t] = lambda _: _raise(
            NotImplementedError
        ),  # Default to raise NotImplemented
        msg_type=FeatureLabelPair,
        data_package_limit: int = 100,
    ):
        super().__init__("flower_subscriber")
        FlowerSubscriber.count += 1
        self.nid: int = FlowerSubscriber.count
        self.subscription: Subscription = self.create_subscription(
            msg_type=msg_type,
            topic="test_tensor_topic",
            callback=self.listener_callback,
        )
        self.flower_hook_fn: RosClient.flower_hook_t = flower_hook_fn
        self.data_package_limit: int = data_package_limit
        self.feature_label_pairs: List[Tuple[Tensor, Tensor]] = []
        fl.client.start_client(
            server_address="[::]:8080",
            client=RosClient(self.nid, self, self.data_package_limit).to_client(),
        )

    def set_flower_hook(self, flower_hook_fn: RosClient.flower_hook_t):
        """
        Changes flower hook function
        """
        self.flower_hook = flower_hook_fn

    def listener_callback(self, msg: FeatureLabelPair):
        """
        Callback when subscriber receives broadcast from publisher
        Parameters:
            msg: The message sent by the publisher; assumed to be FeatureLabelPair unless otherwise
        """
        # TODO: link this to be handled in the flower client
        if self.package_size >= self.data_package_limit:
            save_path = path.join(
                self.data_folder, str(datetime.now().timestamp(), ".pb")
            )
            torch.save(
                self.feature_label_pairs,
                f"{save_path}",
            )
            # For message passing between ROS node that flower client;
            # currently it only sends the processed training data package
            # oo the flower client
            try:
                self.flower_hook(save_path)
            except NotImplementedError:
                raise NotImplementedError(
                    "Flower hook is required for communication with flower"
                )

            self.feature_label_pairs.clear()
        self.feature_label_pairs.append(msg.torch())
