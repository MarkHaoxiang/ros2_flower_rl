from __future__ import annotations

import torch
import numpy as np

import ml_interfaces.msg as msg

class FloatTensor(msg.FloatTensor):
    """ A wrapper around the FloatTensor msg with utilities
    """

    @staticmethod
    def from_torch(data: torch.Tensor) -> FloatTensor:
        """ Constructs a ROS-compatible FloatTensor from Torch Tensor

        Args:
            data (torch.Tensor): PyTorch Tensor.

        Returns:
            FloatTensor: ROS tensor message.
        """
        return FloatTensor(
            shape = list(data.shape),
            values = data.flatten().to(device="cpu", dtype=torch.float32).tolist()
        )

    @staticmethod    
    def from_numpy(data: np.ndarray) -> FloatTensor:
        """ Constructs a ROS-compatible FloatTensor from ndarray

        Args:
            data (torch.Tensor): Numpy ndarray.

        Returns:
            FloatTensor: ROS tensor message.
        """
        return FloatTensor(
            shape = list(data.shape),
            values = data.flatten().tolist()
        )

    def torch(self, **kwargs) -> torch.Tensor:
        """ Converts to torch tensor.

        Returns:
            torch.Tensor: result.
        """
        tensor =  torch.tensor(data=self.values, dtype=torch.float32, **kwargs)
        tensor = tensor.reshape(self.shape)
        return tensor

    def numpy(self, **kwargs) -> np.ndarray:
        """ Converts to numpy ndarray.

        Returns:
            np.ndarray: results.
        """
        arr = np.array(self.values, dtype=np.float32, **kwargs)
        arr = arr.reshape(self.shape)
        return arr

    def pack(self) -> msg.FloatTensor:
        return msg.FloatTensor(shape=self.shape, values=self.values)
