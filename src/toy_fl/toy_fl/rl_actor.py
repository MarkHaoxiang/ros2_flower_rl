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
    def __init__(self, node_name: str, srv_name: str, replay_buffer_name: str):
        Node.__init__(self, f"Reinforcement Learning Actor {node_name}")
        self.service = self.create_service(
            srv.ControllerService,
            srv_name=srv_name,
            callback=self.on_request_callback,
        )
        self.memory_client = self.create_client(
            srv_type=srv.SampleTransition, srv_name=replay_buffer_name
        )

    @property
    def _action_type(self) -> Type:
        """This hack should (no tested) return the
        instantiated ActionType in derived class"""
        original_bases = type(self).__orig_bases__  # type: ignore
        for base in original_bases:
            if isinstance(base, _GenericAlias):
                concrete_action_type = get_args(base)[0]
                return concrete_action_type
        raise NotImplementedError("State Type is Not Correctly Instantiated")

    @property
    def _state_type(self) -> Type:
        """This hack should (no tested) return the
        instantiated StateType in derived class"""
        original_bases = type(self).__orig_bases__  # type: ignore
        for base in original_bases:
            if isinstance(base, _GenericAlias):
                concrete_action_type = get_args(base)[1]
                return concrete_action_type
        raise NotImplementedError("State Type is Not Correctly Instantiated")

    def on_request_callback(
        self,
        request: srv.ControllerService.Request,
        response: srv.ControllerService.Response,
    ) -> srv.ControllerService.Response:
        obs: StateType = ControllerService.unpack_request(
            request, state_type=self._state_type
        )
        action: ActionType = self.policy(obs)
        ControllerService.set_response(response, action=action)
        return response

    @property
    @abstractmethod
    def policy(self) -> kitten.policy.Policy:
        raise NotImplementedError

    def sample_request(
        self, n: int
    ) -> tuple[kitten.experience.Transition, kitten.experience.AuxiliaryMemoryData]:
        """Samples a batch from memory

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
            indices=None,
        )
        return batch, aux
