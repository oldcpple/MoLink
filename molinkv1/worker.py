"""A GPU worker class."""

import gc
import os
from types import NoneType
from typing import TYPE_CHECKING, Optional

import torch
from vllm.config import VllmConfig
from vllm.distributed import (
    ensure_model_parallel_initialized,
    init_distributed_environment,
    set_custom_all_reduce,
)
from vllm.distributed.ec_transfer import ensure_ec_transfer_initialized
from vllm.distributed.parallel_state import (
    get_pp_group,
    get_tp_group,
)
from vllm.logger import init_logger
from vllm.model_executor import set_random_seed
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.utils.mem_constants import GiB_bytes
from vllm.utils.mem_utils import MemorySnapshot
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.utils import report_usage_stats
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.v1.worker.utils import is_residual_scattered_for_sp
from vllm.v1.worker.gpu_worker import Worker

logger = init_logger(__name__)

if TYPE_CHECKING:
    from vllm.model_executor.model_loader.tensorizer import TensorizerConfig
from molinkv1.scheduler import MolinkSchedulerOutput

class MolinkWorker(Worker):
    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
    ):
        super().__init__(
            vllm_config=vllm_config,
            local_rank=local_rank,
            rank=rank,
            distributed_init_method=distributed_init_method,
            is_driver_worker=is_driver_worker,
        )
        # For MoLink: storage for intermediate tensors from gRPC
        self._molink_intermediate_tensors: Optional[IntermediateTensors] = None

    def _molink_set_intermediate_tensors(
        self, intermediate_tensors: IntermediateTensors
    ) -> None:
        """Set intermediate tensors for MoLink (called via RPC).

        Args:
            intermediate_tensors: Tensors from previous pipeline stage.
        """
        self._molink_intermediate_tensors = intermediate_tensors
        # logger.info(
        #     f"[MoLink][Worker] Set intermediate tensors with {len(intermediate_tensors.tensors)} items"
        # )

    def _molink_get_intermediate_tensors(self) -> Optional[IntermediateTensors]:
        """Get and clear stored intermediate tensors for MoLink.

        Returns:
            Stored intermediate tensors or None.
        """
        tensors = self._molink_intermediate_tensors
        self._molink_intermediate_tensors = None
        return tensors

    def init_device(self):
        if self.device_config.device.type == "cuda":
            # This env var set by Ray causes exceptions with graph building.
            os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)
            if (
                self.parallel_config.data_parallel_size > 1
                and self.parallel_config.data_parallel_size_local > 0
                and self.parallel_config.distributed_executor_backend
                not in ["ray", "external_launcher"]
                and self.vllm_config.parallel_config.data_parallel_backend != "ray"
                and self.vllm_config.parallel_config.nnodes_within_dp == 1
            ):
                # Use local DP rank if available, otherwise use global DP rank.
                dp_local_rank = self.parallel_config.data_parallel_rank_local
                if dp_local_rank is None:
                    dp_local_rank = self.parallel_config.data_parallel_rank

                tp_pp_world_size = (
                    self.parallel_config.pipeline_parallel_size
                    * self.parallel_config.tensor_parallel_size
                )

                # DP_LOCAL_RANK * TP_PP_WORLD_SIZE + TP_LOCAL_RANK
                self.local_rank += dp_local_rank * tp_pp_world_size
                assert (
                    self.local_rank < torch.cuda.device_count()
                ), f"DP adjusted local rank {self.local_rank} is out of bounds. "
                visible_device_count = (
                    torch.cuda.device_count() if torch.cuda.is_available() else 0
                )
                assert self.parallel_config.local_world_size <= visible_device_count, (
                    f"local_world_size ({self.parallel_config.local_world_size}) must "
                    f"be less than or equal to the number of visible devices "
                    f"({visible_device_count})."
                )
            self.device = torch.device(f"cuda:{self.local_rank}")
            current_platform.set_device(self.device)

            current_platform.check_if_supports_dtype(self.model_config.dtype)
            # Initialize the distributed environment BEFORE taking
            # memory snapshot
            # This ensures NCCL buffers are allocated before we measure
            # available memory
            init_worker_distributed_environment(
                self.vllm_config,
                self.rank,
                self.distributed_init_method,
                self.local_rank,
                current_platform.dist_backend,
            )

            # Set random seed.
            set_random_seed(self.model_config.seed)

            # Now take memory snapshot after NCCL is initialized
            gc.collect()
            torch.cuda.empty_cache()

            # take current memory snapshot
            self.init_snapshot = MemorySnapshot()
            self.requested_memory = (
                self.init_snapshot.total_memory
                * self.cache_config.gpu_memory_utilization
            )
            if self.init_snapshot.free_memory < self.requested_memory:
                GiB = lambda b: round(b / GiB_bytes, 2)
                raise ValueError(
                    f"Free memory on device "
                    f"({GiB(self.init_snapshot.free_memory)}/"
                    f"{GiB(self.init_snapshot.total_memory)} GiB) on startup "
                    f"is less than desired GPU memory utilization "
                    f"({self.cache_config.gpu_memory_utilization}, "
                    f"{GiB(self.requested_memory)} GiB). Decrease GPU memory "
                    f"utilization or reduce GPU memory used by other processes."
                )
        else:
            raise RuntimeError(f"Not support device type: {self.device_config.device}")

        # Construct the model runner
        self.model_runner: GPUModelRunner = GPUModelRunner(
            self.vllm_config, self.device
        )

        if self.rank == 0:
            # If usage stat is enabled, collect relevant info.
            report_usage_stats(self.vllm_config)

    @torch.inference_mode()
    def execute_model(
        self, scheduler_output: "MolinkSchedulerOutput"
    ) -> ModelRunnerOutput | None:
        intermediate_tensors = None
        forward_pass = scheduler_output.total_num_scheduled_tokens > 0
        num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        num_input_tokens = self.model_runner._get_num_input_tokens(num_scheduled_tokens)
        all_gather_tensors = {
            "residual": not is_residual_scattered_for_sp(
                self.vllm_config, num_input_tokens
            )
        }

        # Check if MoLink is enabled
        is_molink = (
            hasattr(self.vllm_config, "molink_config")
            and self.vllm_config.molink_config
            and self.vllm_config.molink_config.enabled
        )

        if forward_pass and not get_pp_group().is_first_rank:
            if is_molink:
                # For MoLink: get intermediate tensors from local storage
                intermediate_tensors = self._molink_get_intermediate_tensors()
                if not intermediate_tensors:
                    logger.warning(
                        f"[MoLink][Worker] No intermediate tensors found in local storage!"
                    )
                # else:
                #     logger.info(
                #         f"[MoLink][Worker] Retrieved intermediate tensors with {len(intermediate_tensors.tensors)} items"
                #     )
            else:
                # Standard NCCL pipeline parallel
                intermediate_tensors = IntermediateTensors(
                    get_pp_group().recv_tensor_dict(
                        all_gather_group=get_tp_group(),
                        all_gather_tensors=all_gather_tensors,
                    )
                )

        with self.annotate_profile(scheduler_output):
            output = self.model_runner.execute_model(
                scheduler_output, intermediate_tensors
            )
            if isinstance(output, (ModelRunnerOutput, NoneType)):
                return output

        assert isinstance(output, IntermediateTensors)
        parallel_config = self.vllm_config.parallel_config

        if is_molink:
            # For MoLink: check if we're the last stage
            # Last stage should NOT return intermediate tensors here
            # Instead, return None to trigger sampling
            if get_pp_group().is_last_rank:
                # logger.info(
                #     f"[MoLink][Worker] Last stage, returning None to trigger sampling"
                # )
                return None
            else:
                # Non-last stage: intermediate tensors will be sent via gRPC
                # logger.info(
                #     f"[MoLink][Worker] Intermediate stage, generated {len(output.tensors)} tensors, will be sent via gRPC"
                # )
                return output
        else:
            # Standard NCCL pipeline parallel
            assert (
                parallel_config.distributed_executor_backend != "external_launcher"
                and not get_pp_group().is_last_rank
            )

            get_pp_group().send_tensor_dict(
                output.tensors,
                all_gather_group=get_tp_group(),
                all_gather_tensors=all_gather_tensors,
            )

            return None


def init_worker_distributed_environment(
    vllm_config: VllmConfig,
    rank: int,
    distributed_init_method: str | None = None,
    local_rank: int = -1,
    backend: str = "nccl",
) -> None:
    """Initialize the distributed environment."""
    parallel_config = vllm_config.parallel_config
    from vllm.model_executor.layers.batch_invariant import init_batch_invariance

    init_batch_invariance()
    set_custom_all_reduce(not parallel_config.disable_custom_all_reduce)

    init_distributed_environment(
        parallel_config.world_size, rank, distributed_init_method, local_rank, backend
    )

    ensure_model_parallel_initialized(
        parallel_config.tensor_parallel_size,
        parallel_config.pipeline_parallel_size,
        parallel_config.decode_context_parallel_size,
    )

    # Apply MoLink patches after distributed initialization if enabled
    molink_config = getattr(vllm_config, "molink_config", None)
    if molink_config is not None and molink_config.enabled:
        from molinkv1.parallel_state import (
            init_molink_parallel_state,
            apply_molink_patches,
        )

        start_layer, end_layer = molink_config.get_serving_layers()
        init_molink_parallel_state(
            enabled=True,
            start_layer=start_layer,
            end_layer=end_layer,
        )
        apply_molink_patches()
        logger.info(
            f"Worker {rank}: MoLink initialized with layers {start_layer}-{end_layer}"
        )

    # Init ec connector here before KV caches caches init
    # NOTE: We do not init KV caches for Encoder-only instance in EPD disagg mode
    ensure_ec_transfer_initialized(vllm_config)
