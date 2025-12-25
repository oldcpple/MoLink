"""玄学bug，删掉就会导致通信死锁😢"""

from typing import TYPE_CHECKING, Any, Optional
from vllm.config import VllmConfig
from vllm.sequence import IntermediateTensors
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.worker.gpu_worker import Worker

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput


class MolinkWorkerMixin:
    """Mixin class for MoLink worker functionality."""
