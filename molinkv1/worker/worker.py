"""A GPU worker class for MoLink cross-node pipeline parallelism."""

from types import NoneType
from typing import TYPE_CHECKING, Optional

import torch

from vllm.config import VllmConfig
from vllm.distributed.parallel_state import get_pp_group, get_tp_group
from vllm.logger import init_logger
from vllm.sequence import IntermediateTensors
from vllm.v1.outputs import ModelRunnerOutput, AsyncModelRunnerOutput
from vllm.v1.worker.gpu_worker import AsyncIntermediateTensors, Worker

logger = init_logger(__name__)

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput


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
        self._molink_intermediate_tensors: Optional[IntermediateTensors] = None

    def _molink_set_intermediate_tensors(
        self, intermediate_tensors: IntermediateTensors
    ) -> None:
        self._molink_intermediate_tensors = intermediate_tensors

    def _molink_get_intermediate_tensors(self) -> Optional[IntermediateTensors]:
        """Return stored intermediate tensors (kept on current device)."""
        tensors = self._molink_intermediate_tensors
        self._molink_intermediate_tensors = None
        return tensors

    def _molink_get_intermediate_tensors_cpu(self) -> Optional[IntermediateTensors]:
        """Return stored intermediate tensors moved to CPU (for gRPC transfer)."""
        tensors = self._molink_intermediate_tensors
        self._molink_intermediate_tensors = None
        if tensors is not None:
            cpu_tensors = {k: v.cpu() for k, v in tensors.tensors.items()}
            return IntermediateTensors(cpu_tensors)
        return tensors

    def init_device(self):
        import vllm.v1.worker.gpu_worker as gpu_worker_module

        original_init_fn = gpu_worker_module.init_worker_distributed_environment

        def patched_init_worker_distributed_environment(
            vllm_config, rank, distributed_init_method=None, local_rank=-1, backend="nccl"
        ):
            original_init_fn(vllm_config, rank, distributed_init_method, local_rank, backend)
            _apply_molink_distributed_patches(vllm_config, rank)

        gpu_worker_module.init_worker_distributed_environment = (
            patched_init_worker_distributed_environment
        )
        try:
            super().init_device()
        finally:
            gpu_worker_module.init_worker_distributed_environment = original_init_fn

    @torch.inference_mode()
    def execute_model(
        self, scheduler_output: "SchedulerOutput"
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | None:
        if self._pp_send_work:
            for handle in self._pp_send_work:
                handle.wait()
            self._pp_send_work = []

        intermediate_tensors = None
        forward_pass = scheduler_output.total_num_scheduled_tokens > 0

        is_molink = (
            hasattr(self.vllm_config, "molink_config")
            and self.vllm_config.molink_config
            and self.vllm_config.molink_config.enabled
        )

        if forward_pass and not get_pp_group().is_first_rank:
            if is_molink:
                intermediate_tensors = self._molink_get_intermediate_tensors()
                if not intermediate_tensors:
                    logger.warning(
                        "[MoLink][Worker] No intermediate tensors found in local storage!"
                    )
            else:
                tensor_dict, comm_handles, comm_postprocess = (
                    get_pp_group().irecv_tensor_dict(
                        all_gather_group=get_tp_group(),
                    )
                )
                assert tensor_dict is not None
                intermediate_tensors = AsyncIntermediateTensors(
                    tensor_dict,
                    comm_handles=comm_handles,
                    comm_postprocess=comm_postprocess,
                )

        with self.annotate_profile(scheduler_output):
            output = self.model_runner.execute_model(
                scheduler_output, intermediate_tensors
            )
            if isinstance(output, (ModelRunnerOutput, AsyncModelRunnerOutput, NoneType)):
                return output

        assert isinstance(output, IntermediateTensors)

        if is_molink:
            if not get_pp_group().is_last_rank:
                self._molink_intermediate_tensors = output
            return None
        else:
            assert not get_pp_group().is_last_rank
            self._pp_send_work = get_pp_group().isend_tensor_dict(
                output.tensors,
                all_gather_group=get_tp_group(),
                all_gather_tensors={},
            )
            return None


def _apply_molink_distributed_patches(vllm_config: VllmConfig, rank: int):
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
            "Worker %d: MoLink initialized with layers %d-%d",
            rank, start_layer, end_layer,
        )
