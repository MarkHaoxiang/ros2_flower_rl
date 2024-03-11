from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from typing import Generic, Type, TypeVar, _GenericAlias, get_args

import ml_interfaces.srv as srv
import numpy as np
import rclpy
import torch
from ml_interfaces_py import ControllerService
from rclpy.node import Node

ActionType = TypeVar("ActionType", np.ndarray, torch.Tensor)  # ActionType
StateType = TypeVar("StateType", np.ndarray, torch.Tensor)  # StateType


class RlActor(Generic[ActionType, StateType], Node, ABC):
    def __init__(self, node_name: str, controller_name: str, replay_buffer_name: str):
        Node.__init__(self, f"Reinforcement Learning Actor {node_name}")
        self._client = self.create_client(
            srv_type=srv.ControllerService, srv_name=controller_name
        )
        self._memory = self.create_client(
            srv_type=srv.SampleFeatureLabelPair, srv_name=replay_buffer_name
        )

    @property
    def _action_type(self) -> Type:
        """This hack should (no tested) return the
        instantiated ActionType in derived class"""
        original_bases = type(self).__orig_bases__
        for base in original_bases:
            if isinstance(base, _GenericAlias):
                concrete_action_type = get_args(base)[0]
                return concrete_action_type
        raise NotImplementedError("State Type is Not Correctly Instantiated")

    @property
    def _state_type(self) -> Type:
        """This hack should (no tested) return the
        instantiated StateType in derived class"""
        original_bases = type(self).__orig_bases__
        for base in original_bases:
            if isinstance(base, _GenericAlias):
                concrete_action_type = get_args(base)[1]
                return concrete_action_type
        raise NotImplementedError("State Type is Not Correctly Instantiated")

    def action_request(self, action: ActionType) -> StateType:
        try:
            future = self._client.call_async(ControllerService.build_request(action))
        except TypeError:
            raise TypeError(
                "Action Passed in may not have the right type"
                "expect: Union[FloatTensor, torch.Tensor, numpy.NDarray]"
                f"Got {type(action)}"
            )

        try:
            rclpy.spin_until_future_complete(self, future)
            result = future.result()
            assert isinstance(result, srv.ControllerService.Response)
        except Exception as e:
            print(f"An unexpected error occured on controller server's side: {e}")
            sys.exit(1)
        else:
            return ControllerService.unpack_response(result, self._state_type)

    # TODO: Please implement the replay buffen sample
    # Requirement: this should return exactly what
    # kitten.experience.memory.ReplayBuffer.sample()
    # returns
    def sample_request(self, n: int):
        raise NotImplementedError
