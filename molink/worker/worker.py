import gc
import os
from typing import Dict, List, Optional, Set, Tuple, Type, Union
import time
import torch
import traceback
import torch.distributed
from vllm.worker.worker import Worker, _check_if_gpu_supports_dtype
import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.model_executor import set_random_seed
from vllm.distributed import (ensure_kv_transfer_initialized,
                              init_distributed_environment,
                              set_custom_all_reduce)
from vllm.utils import (GiB_bytes, MemorySnapshot, bind_kv_cache,
                        memory_profiling)
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.distributed import get_pp_group
from vllm.worker.model_runner import GPUModelRunnerBase
from vllm.sequence import IntermediateTensors, ExecuteModelRequest
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
        #model_runner_cls = MolinkGPUModelRunner
        super().__init__(vllm_config, local_rank, rank, distributed_init_method, is_driver_worker)
        speculative_config = self.speculative_config
        model_config = self.model_config
        speculative_args = {} if speculative_config is None \
            or (speculative_config.draft_model_config.model ==
                model_config.model) \
            or (speculative_config.draft_model_config.hf_config.model_type
                not in ["medusa", "mlp_speculator", "eagle"]) \
                    else {"return_hidden_states": True}
        self.model_runner: GPUModelRunnerBase = MolinkGPUModelRunner(
            vllm_config=self.vllm_config,
            kv_cache_dtype=self.cache_config.cache_dtype,
            is_driver_worker=is_driver_worker,
            **speculative_args,
        )
        

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
            torch.cuda.reset_peak_memory_stats()
            self.baseline_snapshot = MemorySnapshot()
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

    def execute_model(
        self,
        execute_model_req: Optional[ExecuteModelRequest] = None,
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ):
        """Executes at least one model step on the given sequences, unless no
        sequences are provided."""

        try:
            start_time = time.perf_counter()
            inputs = self.prepare_input(execute_model_req)
            if inputs is None:
                return None

            model_input, worker_input, kwargs = inputs
            num_steps = worker_input.num_steps

            # not surpported in vLLM v0.7.2
            '''
            if (execute_model_req is not None and execute_model_req.spec_step_idx):
                kwargs["spec_step_idx"] = execute_model_req.spec_step_idx
            '''

            self.execute_worker(worker_input)

            # If there is no input, we don't need to execute the model.
            if worker_input.num_seq_groups == 0:
                return []

            orig_model_execute_time = 0.0
            if not get_pp_group().is_first_rank:
                if (self.observability_config is not None
                        and self.observability_config.collect_model_execute_time):
                    orig_model_execute_time = intermediate_tensors.tensors.get(
                        "model_execute_time", torch.tensor(0)).item()

            output = self.model_runner.execute_model(
                model_input=model_input,
                kv_caches=self.kv_cache[worker_input.virtual_engine]
                if self.kv_cache is not None else None,
                intermediate_tensors=intermediate_tensors,
                num_steps=num_steps,
                **kwargs,
            )

            model_execute_time = time.perf_counter() - start_time
            if not get_pp_group().is_last_rank:
                # output is IntermediateTensors
                assert isinstance(output, IntermediateTensors)
                if (self.observability_config is not None
                        and self.observability_config.collect_model_execute_time):
                    output.tensors["model_execute_time"] = torch.tensor(
                        model_execute_time + orig_model_execute_time)
                    
                return [output.tensors]

            if (self.observability_config is not None
                    and self.observability_config.collect_model_execute_time
                    and output is not None):
                for o in output:
                    o.model_execute_time = (orig_model_execute_time +
                                            model_execute_time)

            # output is List[SamplerOutput]
            return output
        
        except Exception as e:
            print(e)
            traceback.print_exc()


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

    # world size in MoLink should be tensor_parallel_size
    init_distributed_environment(parallel_config.tensor_parallel_size, rank,
                                 distributed_init_method, local_rank)
    ensure_model_parallel_initialized(_is_first_rank,
                                      _is_last_rank,
                                      parallel_config.tensor_parallel_size,
                                      parallel_config.pipeline_parallel_size)

    ensure_kv_transfer_initialized(vllm_config)