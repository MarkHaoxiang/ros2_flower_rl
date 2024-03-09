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
                values=data.flatten().to(device="cpu", dtype=torch.float32).tolist(),
            )
        elif isinstance(data, torch.Tensor):
            return FloatTensor(shape=list(data.shape), values=data.flatten().tolist())
        else:
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


class ControllerService(srv.ControllerService):
    @staticmethod
    def build_request(
        action: Union[np.ndarray, torch.Tensor, FloatTensor, msg.FloatTensor],
    ) -> srv.ControllerService.Request:
        """Builds a gym service request

        Args:
            action (np.ndarray | torch.Tensor | FloatTensor | msg.FloatTensor): the action proposed by gym client
            reset (bool): whether this request is a reset request

        -------
        Returns:
            srv.ControllerService.Request The request to be sent to the server
        """
        # TODO: check if we can omit certain fields in the request
        request = super().Request()
        if not isinstance(action, FloatTensor, msg.FloatTensor):
            action = FloatTensor.build(action)
        if not isinstance(action, msg.FloatTensor):
            action = action.pack()
        request.action = action
        return request

    @staticmethod
    def unpack_request(
        request: srv.ControllerService.Request, type: Optional[Type] = None
    ) -> Tuple[Union[FloatTensor, np.ndarray, torch.Tensor], bool]:
        """Unpacks a ControllerService Request
        Args:
            request (srv.ControllerService.Request)

        ------
        Returns:
            action (np.ndarray), reset (bool)
            the action and whether the request is for a reset; if the request is
            a reset, the action would be empty
        """
        action = FloatTensor.unpack(request.action)
        if type is torch.Tensor:
            action = action.torch()
        elif type is np.ndarray:
            action = action.numpy()
        return action

    @staticmethod
    def set_response(
        response: srv.ControllerService.Response,
        s_1: Union[msg.FloatTensor, FloatTensor, torch.Tensor, np.ndarray],
        info: Optional[FloatTensor] = None,
    ) -> None:
        """Sets the content of a ControllerService response

        Args:
            response (srv.ControllerService.Response): The response to edit
            s_1 (FloatTensor | torch.Tensor | np.ndarray): The returned state after action

        -------
        Returns:
            srv.ControllerService.Response The request to be sent to the server
        """
        # TODO: check if we can omit certain fields in the request

        # Convert float tensor input to message float tensor
        if not isinstance(s_1, FloatTensor, msg.FloatTensor):
            s_1 = FloatTensor.build(s_1)
        if not isinstance(s_1, msg.FloatTensor):
            s_1 = s_1.pack()

        response.s_1 = s_1
        return response

    @staticmethod
    def build_response(*args, **kwargs) -> srv.ControllerService.Response:
        """Builds a gym service response

        Since we are using callbacks, this is probably unused

        Args:
            s_1 (FloatTensor | torch.Tensor | np.ndarray): The returned state after action

        -------
        Returns:
            srv.ControllerService.Response The request to be sent to the server
        """
        response: srv.ControllerService.Response = super().Response()
        ControllerService.set_response(response, *args, **kwargs)
        return response

    @staticmethod
    def unpack_response(
        response: srv.ControllerService.Response, state_type: Optional[Type] = None
    ) -> Tuple[Union[FloatTensor, torch.Tensor, np.ndarray], FloatTensor]:
        """Unpacks response from GymServer

        Args:
            response (srv.ControllerService.Response): response from the server
            state_type (Optional Type): Type of desired return value for the state

        -------
        Returns:
            state (FloatTensor, Tensor or NDArray depending on state type),
        """
        s_1 = FloatTensor.unpack(response.s_1)
        if state_type is not None:
            if state_type is torch.Tensor:
                s_1 = s_1.torch()
            elif state_type is np.ndarray:
                s_1 = s_1.numpy()
            else:
                raise NotImplementedError
        return s_1
