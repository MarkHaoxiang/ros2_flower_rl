from __future__ import annotations
from abc import abstractmethod, ABC


import rclpy
import flwr as fl
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    GetPropertiesIns,
    GetPropertiesRes
)

class RosFlowerNode(rclpy.Node, fl.client.Client, ABC):
    def __init__(self,
                 server_addr: str = "[::]8080",
                 *args, **kwargs
        ):
        rclpy.Node.__init__(self, *args, **kwargs)
        fl.client.Client.__init__(self)
        self._server_addr = server_addr

    def flwr_get_properties(self, ins: GetPropertiesIns) -> GetPropertiesRes:
        return super().get_properties(ins)
    
    def flwr_get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        return super().get_parameters(ins)
    
    def fit(self, ins: FitIns) -> FitRes:
        return super().fit(ins)

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        return super().evaluate(ins)

    @abstractmethod
    def start_client(self, *args, **kwargs):
        raise NotImplementedError
