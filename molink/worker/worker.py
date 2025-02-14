import gc
import os
from typing import Dict, List, Optional, Set, Tuple, Type, Union

import torch
import torch.distributed
from vllm.worker.worker import Worker, _check_if_gpu_supports_dtype
import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.model_executor import set_random_seed
from vllm.distributed import (ensure_kv_transfer_initialized,
                              init_distributed_environment,
                              set_custom_all_reduce)
from molink.distributed.parallel_state import ensure_model_parallel_initialized
from molink.worker.model_runner import MolinkGPUModelRunner

class MolinkWorker(Worker):

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
        model_runner_cls: Optional[Type[GPUModelRunnerBase]] = None,
    ) -> None:
        
        '''for current version'''
        model_runner_cls = Type(MolinkGPUModelRunner)
        super().__init__(vllm_config, local_rank, rank, distributed_init_method, is_driver_worker, model_runner_cls)
        


    def init_device(self, _is_first_rank: bool, _is_last_rank: bool,) -> None:
        if self.device_config.device.type == "cuda":
            # torch.distributed.all_reduce does not free the input tensor until
            # the synchronization point. This causes the memory usage to grow
            # as the number of all_reduce calls increases. This env var disables
            # this behavior.
            # Related issue:
            # https://discuss.pytorch.org/t/cuda-allocation-lifetime-for-inputs-to-distributed-all-reduce/191573
            os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"

            # This env var set by Ray causes exceptions with graph building.
            os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)
            self.device = torch.device(f"cuda:{self.local_rank}")
            torch.cuda.set_device(self.device)

            _check_if_gpu_supports_dtype(self.model_config.dtype)
            gc.collect()
            torch.cuda.empty_cache()
            self.init_gpu_memory = torch.cuda.mem_get_info()[0]
        else:
            raise RuntimeError(
                f"Not support device type: {self.device_config.device}")
        # Initialize the distributed environment.
        init_worker_distributed_environment(_is_first_rank,
                                            _is_last_rank,
                                            self.vllm_config, self.rank,
                                            self.distributed_init_method,
                                            self.local_rank)
        # Set random seed.
        set_random_seed(self.model_config.seed)


def init_worker_distributed_environment(
    _is_first_rank: bool,
    _is_last_rank: bool,
    vllm_config: VllmConfig,
    rank: int,
    distributed_init_method: Optional[str] = None,
    local_rank: int = -1,
) -> None:
    """Initialize the distributed environment."""
    parallel_config = vllm_config.parallel_config
    set_custom_all_reduce(not parallel_config.disable_custom_all_reduce)

    init_distributed_environment(parallel_config.world_size, rank,
                                 distributed_init_method, local_rank)
    ensure_model_parallel_initialized(_is_first_rank,
                                      _is_last_rank,
                                      parallel_config.tensor_parallel_size,
                                      parallel_config.pipeline_parallel_size)

    ensure_kv_transfer_initialized(vllm_config)