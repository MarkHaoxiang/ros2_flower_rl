# import rclpy
from rclpy.node import Node
from ml_interfaces_py.lib import FloatTensor, FeatureLabelPair
from typing import Callable, List, Tuple, Optional

import torch
from torch import Tensor
from os import path
from datatime import datetime


def _raise(ex: Exception):
    """Wrapper function to raise exceptions in lambda"""
    raise ex


class FlowerSubscriber(Node):
    """
    ROS node responsible for feeding data sets to flower client,
    communicates with the flower client via a hook function
    """

    flower_hook_t = Callable[str]
    flower_hook: flower_hook_t
    feature_label_pairs: List[Tuple[Tensor, Tensor]] = []
    package_size: int = 0
    data_package_limit: int = 100  # 100 training samples per batch
    data_folder: str = path.join(path.dirname(path.abspath(__file__)), "data")

    def __init__(
        self,
        flower_hook_fn: Optional[Callable] = lambda: _raise(
            NotImplementedError
        ),  # Default to raise NotImplemented
        msg_type=FeatureLabelPair,
        data_package_limit: int = 100,
    ):
        super().__init__("flower_subscriber")
        self.subscriptions = self.create_subscription(
            msg_type=msg_type,
            topic="test_tensor_topic",
            callback=self.listener_callback,
        )
        self.subscription
        self.external_callback = flower_hook_fn
        self.data_package_limit = data_package_limit

    def set_flower_hook(self, flower_hook_fn: flower_hook_t):
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
