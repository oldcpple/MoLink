from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, cast

from pydantic import ConfigDict

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
    initial_peer: Optional[str] = None
    grpc_port: int = 0
    start_layer: int = 0
    end_layer: int = -1
    max_message_size_mb: int = 200
    enable_metrics: bool = False
    max_concurrent_batches: int = 2

    @property
    def enabled(self) -> bool:
        """Auto-detect: MoLink is enabled when not serving all layers on a single node."""
        return not (self.start_layer == 0 and self.end_layer == -1
                    and not self.initial_peer)

    def __post_init__(self):
        if self.max_message_size_mb <= 0:
            raise ValueError("max_message_size_mb must be positive")

    @property
    def is_head_node(self) -> bool:
        return self.initial_peer is None or self.initial_peer == ""

    @property
    def max_message_size_bytes(self) -> int:
        return self.max_message_size_mb * 1024 * 1024

    def get_serving_layers(self) -> tuple[int, int]:
        return (self.start_layer, self.end_layer)


@config(config=ConfigDict(arbitrary_types_allowed=True))
class VllmConfig1(VllmConfig):
    molink_config: MolinkConfig = field(default_factory=MolinkConfig)

    def _update_attr(self, molink_config: MolinkConfig):
        self.molink_config = molink_config


@config
class MolinkSchedulerConfig(SchedulerConfig):
    def get_scheduler_cls(self) -> type["SchedulerInterface"]:
        if self.scheduler_cls is None:
            if self.async_scheduling:
                from molinkv1.core.scheduler import MolinkAsyncScheduler

                return MolinkAsyncScheduler
            from molinkv1.core.scheduler import MolinkScheduler

            return MolinkScheduler

        logger.warning_once(
            "Using custom scheduler class %s. This scheduler interface is "
            "not public and compatibility may not be maintained.",
            self.scheduler_cls,
        )
        if not isinstance(self.scheduler_cls, str):
            return cast(type["SchedulerInterface"], self.scheduler_cls)
        return resolve_obj_by_qualname(self.scheduler_cls)
