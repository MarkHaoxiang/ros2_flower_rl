from threading import Thread

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

from .interface import RosFlowerNode

class _RosFlowerClientProxy(fl.client.Client):
    """ Compatibility layer between ROS and Flower
    """
    def __init__(self,
                 client: RosFlowerNode,
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


class DualThreadClient(RosFlowerNode):
    """ Runs start_client on a separate thread to avoid deadlocks
    """
    def start_client(self, **kwargs):
        client_proxy = _RosFlowerClientProxy(self, self._server_addr)
        def _start_client():
            """ Starts looping client_proxy to communicate with the server
            """
            fl.client.start_client(
                server_address=client_proxy._server_addr,
                client=client_proxy,
                **kwargs
            )
        Thread(target=_start_client).start()
