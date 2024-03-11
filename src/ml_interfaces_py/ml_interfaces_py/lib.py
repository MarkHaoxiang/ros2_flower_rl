from __future__ import annotations
from typing import Union, Tuple, Optional, Type

import torch
import numpy as np

import ml_interfaces.msg as msg
import ml_interfaces.srv as srv

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

    @staticmethod
    def unpack(msg: msg.FeatureLabelPair):
        return FeatureLabelPair(
            feature=FloatTensor.unpack(msg.feature), label=FloatTensor.unpack(msg.label)
        )

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
                values=data.flatten().tolist(),
            )
        elif isinstance(data, torch.Tensor):
            return FloatTensor(shape=list(data.shape), values=data.flatten().tolist())
        else:
            try:
                return FloatTensor.build(np.array(data))
            except:
                raise NotImplementedError

    @staticmethod
    def unpack(msg: msg.FloatTensor) -> FloatTensor:
        return FloatTensor(shape=msg.shape, values=msg.values)

    def torch(self, **kwargs) -> torch.Tensor:
        """Converts to torch tensor.

        Returns:
            torch.Tensor: result.
        """
        tensor = torch.tensor(data=self.values, dtype=torch.float32, **kwargs)
        tensor = tensor.reshape(self.shape.tolist())
        return tensor

    def numpy(self, **kwargs) -> np.ndarray:
        """Converts to numpy ndarray.

        Returns:
            np.ndarray: results.
        """
        arr = np.array(self.values, dtype=np.float32, **kwargs)
        arr = arr.reshape(self.shape.tolist())
        return arr

    def pack(self) -> msg.FloatTensor:
        return msg.FloatTensor(shape=self.shape, values=self.values)


class Transition(msg.Transition):
    """A wrapper around the Transition msg with utilities"""

    @staticmethod
    def build(data: tuple[Tensor, Tensor, float, Tensor, bool]) -> Transition:
        """Constructs a ROS compatible Transition object from data

        Args:
            data (tuple[Tensor, Tensor, float, Tensor, bool]): Transition of
                (state, action, reward, next_state, done)

        Returns:
            Transition: ROS Transition message.
        """
        return Transition(
            s_0=FloatTensor.build(data[0]),
            a=FloatTensor.build(data[1]),
            r=data[2],
            s_1=FloatTensor.build(data[3]),
            d=data[4],
        )

    def torch(
        self, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor, float, torch.Tensor, bool]:
        return (
            self.s_0.torch(**kwargs),
            self.a.torch(**kwargs),
            self.r,
            self.s_1.torch(**kwargs),
            self.d,
        )

    def numpy(self, **kwargs):
        return (
            self.s_0.numpy(**kwargs),
            self.a.numpy(**kwargs),
            self.r,
            self.s_1.numpy(**kwargs),
            self.d,
        )

    @staticmethod
    def unpack(msg: msg.Transition) -> Transition:
        return Transition(
            s_0=FloatTensor.unpack(msg.s_0),
            a=FloatTensor.unpack(msg.a),
            r=msg.r,
            s_1=FloatTensor.unpack(msg.a),
            d=msg.d,
        )

    def pack(self) -> msg.Transition:
        return msg.Transition(
            s_0=self.s_0.pack(),
            a=self.a.pack(),
            r=self.r,
            s_1=self.s_1.pack(),
            d=self.d,
        )
