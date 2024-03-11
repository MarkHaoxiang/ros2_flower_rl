from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from typing import Generic, Type, TypeVar, _GenericAlias, get_args

import rclpy
from rclpy.node import Node
import ml_interfaces.srv as srv
import ml_interfaces.msg as msg
import numpy as np
import torch
import kitten


from ml_interfaces_py import Transition, ControllerService

ActionType = TypeVar("ActionType", np.ndarray, torch.Tensor)  # ActionType
StateType = TypeVar("StateType", np.ndarray, torch.Tensor)  # StateType


class RlActor(Generic[ActionType, StateType], Node, ABC):
    def __init__(self, node_name: str, controller_name: str, replay_buffer_name: str):
        Node.__init__(self, f"Reinforcement Learning Actor {node_name}")
        self._client = self.create_client(
            srv_type=srv.ControllerService, srv_name=controller_name
        )
        self.memory_client = self.create_client(
            srv_type=srv.SampleTransition, srv_name=replay_buffer_name
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

    def sample_request(self, n: int) -> tuple[kitten.experience.Transition, kitten.experience.AuxiliaryMemoryData]:
        """ Samples a batch from memory

        Args:
            n (int): minibatch size.

        Returns:
            tuple[kitten.experience.Transition, kitten.experience.AuxiliaryMemoryData]: training batch
        """
        request = srv.SampleTransition.Request(n=n)
        future = self.memory_client.call_async(request)
        rclpy.spin_until_future_complete(future)
        batch = [Transition.unpack(x) for x in future.result().batch]
        batch = [x.numpy() for x in batch]
        batch = kitten.experience.util.build_transition_from_list(batch)
        aux = kitten.experience.AuxiliaryMemoryData(
            weights=torch.ones(len(batch.s_0), batch.s_0.get_device()),
            random=None,
            indices=None
        )
        return batch, aux

