import argparse
import dataclasses
from dataclasses import dataclass
from typing import Optional
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.utils.argparse_utils import FlexibleArgumentParser


@dataclass
class MolinkEngineArgs(AsyncEngineArgs):
    # MoLink-specific fields
    molink_enabled: bool = False
    molink_initial_peer: Optional[str] = None
    molink_grpc_port: int = 0
    molink_start_layer: int = 0
    molink_end_layer: int = -1
    molink_max_message_size_mb: int = 200
    molink_connection_timeout_s: float = 30.0
    molink_heartbeat_interval_s: float = 5.0
    molink_enable_compression: bool = False
    molink_num_delivery_workers: int = 2

    @classmethod
    def add_cli_args(cls, parser: FlexibleArgumentParser) -> FlexibleArgumentParser:
        parser = super(MolinkEngineArgs, cls).add_cli_args(parser)
        # Individual MoLink arguments
        parser.add_argument(
            "--molink-enabled",
            action="store_true",
            help="Enable MoLink cross-node pipeline parallelism.",
        )
        parser.add_argument(
            "--molink-initial-peer",
            type=str,
            default=None,
            help="The gRPC address of the initial node to join (e.g., '192.168.1.100:50051'). If not provided, this node is the head node.",
        )
        parser.add_argument(
            "--molink-grpc-port",
            type=int,
            default=0,
            help="The gRPC port to listen on. If 0, a free port will be chosen automatically.",
        )
        parser.add_argument(
            "--molink-start-layer",
            type=int,
            default=0,
            help="The first layer this node handles (inclusive).",
        )
        parser.add_argument(
            "--molink-end-layer",
            type=int,
            default=-1,
            help="The last layer this node handles (exclusive). -1 means all remaining layers.",
        )
        parser.add_argument(
            "--molink-max-message-size-mb",
            type=int,
            default=200,
            help="Maximum gRPC message size in MB.",
        )
        parser.add_argument(
            "--molink-connection-timeout-s",
            type=float,
            default=30.0,
            help="Timeout for gRPC connections in seconds.",
        )
        parser.add_argument(
            "--molink-heartbeat-interval-s",
            type=float,
            default=5.0,
            help="Interval for health check heartbeats in seconds.",
        )
        parser.add_argument(
            "--molink-enable-compression",
            action="store_true",
            help="Enable gRPC message compression.",
        )
        parser.add_argument(
            "--molink-num-delivery-workers",
            type=int,
            default=2,
            help="Number of workers for async tensor delivery.",
        )
        return parser

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        # Get the list of attributes of this dataclass.
        attrs = [attr.name for attr in dataclasses.fields(cls)]
        # Set the attributes from the parsed arguments.
        engine_args = cls(
            **{attr: getattr(args, attr) for attr in attrs if hasattr(args, attr)}
        )
        return engine_args

