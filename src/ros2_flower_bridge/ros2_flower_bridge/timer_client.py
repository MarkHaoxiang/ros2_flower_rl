from typing import Callable

from flwr.client.app import _check_actionable_client, _init_connection
from flwr.client.client_app import ClientApp
from flwr.client.message_handler.message_handler import handle_control_message
from flwr.client.node_state import NodeState
from flwr.common import GRPC_MAX_MESSAGE_LENGTH
from flwr.common.retry_invoker import RetryInvoker, exponential
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup 

from .interface import RosFlowerNode

class TimerCallbackClient(RosFlowerNode):
    def __init__(self, *args, server_addr: str = "[::]:8080", **kwargs):
        super().__init__(*args, server_addr=server_addr, **kwargs)
        self._flower_cbg = MutuallyExclusiveCallbackGroup()

    """ Uses timer callbacks to manage communication to the server

    If possible, use this instead of DualThreadClient
    - Avoids spinning on a loop to conserve resources

    #TODO: Support async
    """
    def start_client(
        self,
        grpc_max_message_length: int = GRPC_MAX_MESSAGE_LENGTH,
        root_certificates: bytes | str | None = None,
        insecure: bool | None = None,
        transport: str | None = None,
        max_retries: int | None = None,
        max_wait_time: float | None = None,
    ):
        _check_actionable_client(self._client_proxy, None)
        client_fn = lambda _: self._client_proxy

        if insecure is None:
            insecure = root_certificates is None

        load_client_app_fn = lambda: ClientApp(client_fn=client_fn)
        self.connection, address, connection_error_type = _init_connection(
            transport, self._server_addr
        )

        retry_invoker = RetryInvoker(
            wait_factory=exponential,
            recoverable_exceptions=connection_error_type,
            max_tries=max_retries,
            max_time=max_wait_time,
            # TODO: add handlers for success, failure and setback
        )

        self.node_state = NodeState()
        self._training_ended = False
        self._train_timer = None
        self.client_manager_timer = self.create_timer(
            timer_period_sec=1.0,
            callback=lambda: self.flwr_timed_connect_callback(
                address=address,
                insecure=insecure,
                root_certificates=root_certificates,
                retry_invoker=retry_invoker,
                grpc_max_message_length=grpc_max_message_length,
                load_client_app_fn=load_client_app_fn,
            ),
            callback_group=self._flower_cbg,
        )
        self.get_logger().info("Client Manager Timer started")

    def flwr_timed_connect_callback(
        self,
        address: str,
        insecure: bool,
        root_certificates,
        retry_invoker,
        grpc_max_message_length,
        load_client_app_fn: Callable[[], ClientApp],
    ):
        if _timer_running(self._train_timer) and not self._training_ended:        
            self.get_logger().info("Client Manager callback not needed")
            return

        self.get_logger().info("Client Manager callback called")
        if not _timer_running(self._train_timer) and not self._training_ended:
            self.conn = self.connection(address, insecure, retry_invoker, grpc_max_message_length, root_certificates)
            receive, send, create_node, delete_node = self.conn.__enter__()
            # Register node
            if create_node is not None:
                create_node()  # pylint: disable=not-callable
            if self._train_timer is not None:
                self._train_timer.reset()
            else:
                self._train_timer = self.create_timer(
                    timer_period_sec=3.0,
                    callback=lambda: self.flwr_timed_message_callback(
                        receive=receive, send=send, load_client_app_fn=load_client_app_fn
                    ),
                    callback_group=self._flower_cbg
                )
        elif _timer_running(self._train_timer) and self._training_ended:
            self.conn.__exit__(None, None, None)
            assert (
                self._train_timer is not None
            )  # Unneccesary, trying to get the linter happy
            self._train_timer.cancel()
            self.get_logger().info("Training stopped by control message")
        else:
            self.get_logger().info("Training has been suspended or terminated")

    def flwr_timed_message_callback(self, receive, send, load_client_app_fn):
        message = receive()
        if message is None:
            return
        self.get_logger().info(
            f"[RUN { message.metadata.run_id }, ROUND { message.metadata.group_id }]"
        )
        self.get_logger().info(
            f"Received: { message.metadata.message_type } message { message.metadata.message_id }"
        )
        # Handle control message
        out_message, _ = handle_control_message(message)
        if out_message:
            send(out_message)
            self._training_ended = True
            return
        # Register context for this run
        self.node_state.register_context(run_id=message.metadata.run_id)
        # Retrieve context for this run
        context = self.node_state.retrieve_context(run_id=message.metadata.run_id)
        client_app: ClientApp = load_client_app_fn()
        out_message = client_app(message=message, context=context)
        self.get_logger().info("Out message is ready")
        # Update node state
        self.node_state.update_context(
            run_id=message.metadata.run_id,
            context=context,
        )

        # Send
        send(out_message)
        self.get_logger().info(
            f"Sent: {out_message.metadata.message_type} reply to message {message.metadata.message_id}"
        )

def _timer_running(timer):
    return timer is not None and not timer.is_canceled()
