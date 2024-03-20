from threading import Thread

import flwr as fl
from .interface import RosFlowerNode, RosFlowerClientProxy

class DualThreadClient(RosFlowerNode):
    """ Runs start_client on a separate thread to avoid deadlocks
    """
    def start_client(self, **kwargs):
        def _start_client():
            """ Starts looping client_proxy to communicate with the server
            """
            fl.client.start_client(
                server_address=self._server_addr,
                client=self._client_proxy
                **kwargs
            )
        Thread(target=_start_client, daemon=True).start()
