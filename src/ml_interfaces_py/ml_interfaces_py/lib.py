from __future__ import annotations

import torch
import numpy as np

import ml_interfaces.msg as msg

# Typing utility
Tensor = torch.Tensor | np.ndarray


class FeatureLabelPair(msg.FeatureLabelPair):
    """A wrapper around the FeatureLabelPair msg with utilities"""

    @staticmethod
    def build(
        feature: Tensor | FloatTensor, label: Tensor | FloatTensor
    ) -> FeatureLabelPair:
        """
        Constructs a ROS-compatible FeatureLabelPair from input data

        Args:
            feature (Tensor | FloatTensor): Tensor.
            label (Tensor | FloatTensor): Tensor.

        Returns:
            FeatureLabelPair: ROS FeatureLabelPair message.
        """
        if not isinstance(feature, FloatTensor):
            feature = FloatTensor.build(feature)
        if not isinstance(label, FloatTensor):
            label = FloatTensor.build(label)
        return FeatureLabelPair(feature=feature, label=label)

    def torch(self, **kwargs):
        return (self.feature.torch(**kwargs), self.label.torch(**kwargs))

    def numpy(self, **kwargs):
        return (self.feature.numpy(**kwargs), self.label.numpy(**kwargs))

    def pack(self) -> msg.FeatureLabelPair:
        return msg.FeatureLabelPair(
            feature=self.feature.pack(), label=self.label.pack()
        )


class FloatTensor(msg.FloatTensor):
    """A wrapper around the FloatTensor msg with utilities"""

    @staticmethod
    def build(data: torch.Tensor | np.ndarray) -> FloatTensor:
        """Constructs a ROS-compatible FloatTensor from input data

        Args:
            data (torch.Tensor | np.ndarray): Tensor.

        Raises:
            NotImplementedError: Input tensor type not accepted

        Returns:
            FloatTensor: ROS tensor message.
        """
        if isinstance(data, np.ndarray):
            return FloatTensor(
                shape=list(data.shape),
                values=data.flatten().to(device="cpu", dtype=torch.float32).tolist(),
            )
        elif isinstance(data, torch.Tensor):
            return FloatTensor(shape=list(data.shape), values=data.flatten().tolist())
        else:
            raise NotImplementedError

    def torch(self, **kwargs) -> torch.Tensor:
        """Converts to torch tensor.

        Returns:
            torch.Tensor: result.
        """
        tensor = torch.tensor(data=self.values, dtype=torch.float32, **kwargs)
        tensor = tensor.reshape(self.shape)
        return tensor

    def numpy(self, **kwargs) -> np.ndarray:
        """Converts to numpy ndarray.

        Returns:
            np.ndarray: results.
        """
        arr = np.array(self.values, dtype=np.float32, **kwargs)
        arr = arr.reshape(self.shape)
        return arr

    def pack(self) -> msg.FloatTensor:
        return msg.FloatTensor(shape=self.shape, values=self.values)
