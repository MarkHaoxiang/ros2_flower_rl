from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from typing import Generic, Type, TypeVar, _GenericAlias, get_args

import kitten
import ml_interfaces.msg as msg
import ml_interfaces.srv as srv
import numpy as np
import rclpy
import torch
from ml_interfaces_py import FloatTensor, Transition
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.node import Node

ActionType = TypeVar("ActionType", np.ndarray, torch.Tensor)  # ActionType
StateType = TypeVar("StateType", np.ndarray, torch.Tensor)  # StateType


class RlActor(Generic[ActionType, StateType], Node, ABC):
    def __init__(self, node_name: str, policy_service: str, policy_update_topic: str):
        Node.__init__(self, f"Reinforcement Learning Actor {node_name}")
        self._cb_group = MutuallyExclusiveCallbackGroup()
        self.subscription = self.create_subscription(
            msg_type=msg.Knowledge,
            topic=policy_update_topic,
            callback=self.update_policy,
            qos_profile=10,
            callback_group=self._cb_group,
        )
        self.service = self.create_service(
            srv.PolicyService,
            srv_name=policy_service,
            callback=self.on_request_callback,
            callback_group=self._cb_group,
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
        raise NotImplementedError("Action Type is Not Correctly Instantiated")

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

    @abstractmethod
    def update_policy(self, msg: msg.Knowledge) -> None:
        """Updates the policy function from arguments passed in"""
        raise NotImplementedError

    @property
    @abstractmethod
    def policy(self) -> kitten.policy.Policy:
        raise NotImplementedError
