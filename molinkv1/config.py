"""
Configuration for MoLink cross-node pipeline parallelism.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, cast
from vllm.config import VllmConfig
from vllm.config.scheduler import SchedulerConfig

from vllm.config.utils import config
from vllm.logger import init_logger
from vllm.utils.import_utils import resolve_obj_by_qualname

if TYPE_CHECKING:
    from vllm.v1.core.sched.interface import SchedulerInterface
logger = init_logger(__name__)


@dataclass
class MolinkConfig:
    """Configuration for MoLink cross-node pipeline parallelism.

    MoLink enables distributed pipeline parallelism across multiple physical
    nodes using gRPC for communication. Each node can run a subset of the
    model layers, enabling inference of very large models that don't fit
    on a single node.

    Attributes:
        enabled: Whether MoLink is enabled.
        initial_peer: The gRPC address of the initial node to join the pipeline.
                     If None, this node is the head node.
        grpc_port: The gRPC port to listen on. If 0, a free port will be chosen.
        start_layer: The first layer this node handles (inclusive).
        end_layer: The last layer this node handles (exclusive).
        max_message_size_mb: Maximum gRPC message size in MB.
        connection_timeout_s: Timeout for gRPC connections in seconds.
        heartbeat_interval_s: Interval for health check heartbeats.
        enable_compression: Whether to enable gRPC message compression.
        num_delivery_workers: Number of workers for async tensor delivery.
        max_batch_num: Maximum number of micro-batches for pipeline parallelism.
                      More batches can help hide communication latency but use more memory.
        enable_micro_batch: Whether to enable micro-batch processing for overlapping
                           computation and communication.
    """

    enabled: bool = False
    initial_peer: Optional[str] = None
    grpc_port: int = 0  # 0 means auto-select
    start_layer: int = 0
    end_layer: int = -1  # -1 means all remaining layers
    max_message_size_mb: int = 200
    connection_timeout_s: float = 30.0
    heartbeat_interval_s: float = 5.0
    enable_compression: bool = False
    num_delivery_workers: int = 2
    max_batch_num: int = 4  # Maximum number of micro-batches (virtual engines)
    enable_micro_batch: bool = True  # Enable micro-batch processing

    def __post_init__(self):
        if self.max_message_size_mb <= 0:
            raise ValueError("max_message_size_mb must be positive")
        if self.connection_timeout_s <= 0:
            raise ValueError("connection_timeout_s must be positive")
        if self.num_delivery_workers <= 0:
            raise ValueError("num_delivery_workers must be positive")
        if self.max_batch_num < 1:
            raise ValueError("max_batch_num must be at least 1")

    @property
    def is_head_node(self) -> bool:
        """Returns True if this is the head node of the pipeline."""
        return self.initial_peer is None or self.initial_peer == ""

    @property
    def max_message_size_bytes(self) -> int:
        """Returns the maximum message size in bytes."""
        return self.max_message_size_mb * 1024 * 1024

    def get_serving_layers(self) -> tuple[int, int]:
        """Returns the range of layers this node serves."""
        return (self.start_layer, self.end_layer)
    
    def get_optimal_batch_num(self, num_requests: int, num_nodes: int) -> int:
        """Calculate optimal number of micro-batches based on workload.
        
        The goal is to have enough batches to overlap computation with
        communication, reducing pipeline bubbles.
        
        Args:
            num_requests: Total number of active requests.
            num_nodes: Number of nodes in the pipeline.
            
        Returns:
            Optimal number of micro-batches.
        """
        if not self.enable_micro_batch:
            return 1
        
        # Base heuristic: at least 2 batches for any multi-node setup
        if num_nodes <= 1:
            return 1
        
        # With more nodes, we need more batches to fill the pipeline
        # and hide communication latency
        base_batches = max(2, num_nodes)
        
        # Scale based on number of requests
        # More requests = can benefit from more batches
        if num_requests < 4:
            return min(base_batches, 2)
        elif num_requests < 16:
            return min(base_batches, 3)
        else:
            return min(base_batches, self.max_batch_num)



@dataclass
class VllmConfig1(VllmConfig):

    molink_config: MolinkConfig = field(default_factory=MolinkConfig, init=True)

    def _update_attr(self, molink_config: MolinkConfig):
        self.molink_config = molink_config


@config
@dataclass
class MolinkSchedulerConfig(SchedulerConfig):
    def get_scheduler_cls(self) -> type["SchedulerInterface"]:
        if self.scheduler_cls is None:
            if self.async_scheduling:
                from molinkv1.scheduler import MolinkAsyncScheduler

                return MolinkAsyncScheduler
            from molinkv1.scheduler import MolinkScheduler

            return MolinkScheduler

        # This warning can be removed once the Scheduler interface is
        # finalized and we can maintain support for scheduler classes that
        # implement it
        logger.warning_once(
            "Using custom scheduler class %s. This scheduler interface is "
            "not public and compatibility may not be maintained.",
            self.scheduler_cls,
        )
        if not isinstance(self.scheduler_cls, str):
            return cast(type["SchedulerInterface"], self.scheduler_cls)
        return resolve_obj_by_qualname(self.scheduler_cls)
