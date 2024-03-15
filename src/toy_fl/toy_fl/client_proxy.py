from __future__ import annotations

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
import flwr as fl

from threading import Thread

class RosFlowerClient(fl.client.Client):
    """ Interface for ROS nodes providing flower client functionality

    To avoid namespace issues
    """
    def flwr_get_properties(self, ins: GetPropertiesIns) -> GetPropertiesRes:
        return super().get_properties(ins)
    
    def flwr_get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        return super().get_parameters(ins)
    
    def fit(self, ins: FitIns) -> FitRes:
        return super().fit(ins)

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        return super().evaluate(ins)

class RosFlowerClientProxy(fl.client.Client):
    """ Compatibility layer between ROS and Flower
    """
    def __init__(self,
                 client: RosFlowerClient,
                 server_addr: str = "[::]8080") -> None:
        super().__init__()
        self._server_addr = server_addr
        self._client = client

    def get_properties(self, ins: GetPropertiesIns) -> GetPropertiesRes:
        return self._client.flwr_get_properties(ins)

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        return self._client.flwr_get_parameters(ins)

    def fit(self, ins: FitIns) -> FitRes:
        return self._client.fit(ins)

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        return self._client.evaluate(ins)

def start_client(client_proxy: RosFlowerClientProxy, **kwargs):
    def _start_client():
        """ Starts looping client_proxy to communicate with the server
        """
        fl.client.start_client(
            server_address=client_proxy._server_addr,
            client=client_proxy,
            **kwargs
        )
    Thread(target=_start_client).start()