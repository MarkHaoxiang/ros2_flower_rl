from __future__ import annotations
from abc import abstractmethod, ABC


from rclpy.node import Node
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

class RosFlowerClientProxy(fl.client.Client):
    """ Compatibility layer between ROS and Flower
    """
    def __init__(self,
                 client: RosFlowerNode) -> None:
        super().__init__()
        self._client = client

    def get_properties(self, ins: GetPropertiesIns) -> GetPropertiesRes:
        return self._client.flwr_get_properties(ins)

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        return self._client.flwr_get_parameters(ins)

    def fit(self, ins: FitIns) -> FitRes:
        return self._client.fit(ins)

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        return self._client.evaluate(ins)

class RosFlowerNode(Node, fl.client.Client, ABC):
    def __init__(self,
                 *args,
                 server_addr: str = "[::]:8080",
                 **kwargs
        ):
        Node.__init__(self, *args, **kwargs)
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
