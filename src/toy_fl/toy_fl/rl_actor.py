from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from typing import Generic, Type, TypeVar, _GenericAlias, get_args

import rclpy
from rclpy.node import Node
import ml_interfaces.srv as srv
import ml_interfaces.msg as msg
from ml_interfaces_py import FloatTensor
import numpy as np
import torch
import kitten


from ml_interfaces_py import Transition

ActionType = TypeVar("ActionType", np.ndarray, torch.Tensor)  # ActionType
StateType = TypeVar("StateType", np.ndarray, torch.Tensor)  # StateType


class RlActor(Generic[ActionType, StateType], Node, ABC):
    def __init__(self, node_name: str, srv_name: str, replay_buffer_name: str):
        Node.__init__(self, f"Reinforcement Learning Actor {node_name}")
        self.service = self.create_service(
            srv.PolicyService,
            srv_name=srv_name,
            callback=self.on_request_callback,
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
        request: srv.PolicyService.Request,
        response: srv.PolicyService.Response,
    ) -> srv.PolicyService.Response:
        obs = FloatTensor.unpack(request.s_0).torch()
        action: ActionType = self.policy(obs)
        response.a = FloatTensor.build(action).pack()
        return response

    @property
    @abstractmethod
    def policy(self) -> kitten.policy.Policy:
        raise NotImplementedError
