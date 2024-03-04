from os import path
from datetime import datetime

import rclpy
from rclpy.node import Node, Subscription
from typing import List, Tuple, Optional

import torch
from torch import Tensor
import flwr as fl

from ml_interfaces_py import FeatureLabelPair
from .flwr_client import RosClient


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
        """
        ROS node which subscribes to a data source, batch up the data
        and pass to a flower client for FL

        Parameters:
            flower_hook_fn: Optional[RosClient.flower_hook_t] a hook function passing data to the flower client
            msg_type: ROS message type broadcasted by the flower server
            data_package_limit: int  The number of data entries the subscriber node will put into a pb file for storage
        """
        super().__init__("flower_subscriber")
        FlowerSubscriber.count += 1
        self.nid: int = FlowerSubscriber.count
        self.subscription: Subscription = self.create_subscription(
            msg_type=msg_type,
            topic="data_stream",
            callback=self.listener_callback,
            qos_profile=10,
        )
        self.flower_hook_fn: RosClient.flower_hook_t = flower_hook_fn
        self.data_package_limit: int = data_package_limit
        self.feature_label_pairs: List[Tuple[Tensor, Tensor]] = []
        fl.client.start_client(
            server_address="[::]:8080",
            client_fn=lambda: RosClient(
                cid=self.nid,
                subscriber_node=self,
                entries_per_package=self.data_package_limit,
            ).to_client(),
            insecure=True,
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


def main(args=None):
    rclpy.init(args=args)
    node = FlowerSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
