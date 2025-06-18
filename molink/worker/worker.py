import gc
import os
from typing import Dict, List, Optional, Set, Tuple, Type, Union
import time
import torch
import traceback
import dataclasses
import torch.distributed
from vllm.worker.worker import Worker, _check_if_gpu_supports_dtype
import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.model_executor import set_random_seed
from vllm.distributed.kv_transfer import ensure_kv_transfer_initialized
from vllm.distributed import (init_distributed_environment,
                              set_custom_all_reduce,
                              get_pp_group)
from vllm.utils import (GiB_bytes, MemorySnapshot, bind_kv_cache,
                        memory_profiling)
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.distributed import get_pp_group
from vllm.worker.model_runner import GPUModelRunnerBase
from vllm.sequence import IntermediateTensors, ExecuteModelRequest
from vllm.worker.model_runner_base import (BroadcastableModelInput,
                                           ModelRunnerBase,
                                           ModelRunnerInputBase)
from vllm.worker.worker_base import WorkerInput, extract_previous_hidden_states
from vllm.distributed import broadcast_tensor_dict
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
    ) -> Optional[List[SamplerOutput]]:
        """Executes at least one model step on the given sequences, unless no
        sequences are provided."""
        start_time = time.perf_counter()

        inputs = self.prepare_input(execute_model_req, intermediate_tensors)
        if inputs is None:
            return None

        model_input, worker_input, kwargs, intermediate_tensors = inputs
        num_steps = worker_input.num_steps
        if (execute_model_req is not None and execute_model_req.spec_step_idx):
            kwargs["spec_step_idx"] = execute_model_req.spec_step_idx

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
            kv_caches=self.kv_cache[0]
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
    
    def prepare_input(
        self,
        execute_model_req: Optional[ExecuteModelRequest] = None,
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> Optional[Tuple[BroadcastableModelInput, WorkerInput, Dict[
            str, torch.Tensor]]]:
        """
        Prepare the inputs to ModelRunner and workers.
        """
        if self.is_driver_worker:
            if execute_model_req is None and intermediate_tensors is None:
                if self.do_metadata_broadcast:
                    # This signals that there's no more requests to process for
                    # now. All workers are running infinite loop with
                    # broadcast_tensor_dict, and it stops the loop when the
                    # driver broadcasts an empty input. Send an empty input to
                    # notify all other workers to stop their execution loop.
                    broadcast_tensor_dict({}, src=0)
                return None
            return self._get_driver_input_and_broadcast(execute_model_req, intermediate_tensors)
        else:
            return self._get_worker_input_from_broadcast()
        
        
    def _get_driver_input_and_broadcast(
        self, execute_model_req: ExecuteModelRequest, intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> Tuple[BroadcastableModelInput, WorkerInput, Dict[str, torch.Tensor]]:
        """ Get the driver input and broadcast it to other workers.  """
        assert self.is_driver_worker

        worker_input: WorkerInput = self.prepare_worker_input(
            execute_model_req=execute_model_req)
        model_input: ModelRunnerInputBase = (
            self.model_runner.prepare_model_input(
                execute_model_req.seq_group_metadata_list,
                0,
                execute_model_req.finished_requests_ids))

        kwargs = extract_previous_hidden_states(execute_model_req)

        if self.do_metadata_broadcast:
            broadcast_data = worker_input.as_broadcastable_tensor_dict()
            broadcast_data.update(model_input.as_broadcastable_tensor_dict())
            broadcast_data.update(kwargs)
            if intermediate_tensors is not None:
                broadcast_data.update({'intermediate_tensors' : intermediate_tensors.tensors})

            broadcast_tensor_dict(broadcast_data, src=0)

        if execute_model_req.async_callback:
            model_input = dataclasses.replace(  # type: ignore
                model_input,
                async_callback=execute_model_req.async_callback)

        return model_input, worker_input, kwargs, intermediate_tensors
    
    def _get_worker_input_from_broadcast(
        self
    ) -> Optional[Tuple[BroadcastableModelInput, WorkerInput, Dict[str, torch.Tensor]]]:
        """ Get the worker input from the broadcasted tensor dict. """
        assert self.do_metadata_broadcast
        assert not self.is_driver_worker
        broadcast_data = broadcast_tensor_dict(src=0)
        if not broadcast_data:
            return None

        intermediate_tensors = None
        if not get_pp_group().is_first_rank:
            intermediate_tensors = broadcast_data.get('intermediate_tensors')
            new_it = {}
            device = torch.device(f"cuda:{self.local_rank}")
            for key, tensor in intermediate_tensors.items():
                # 确保 tensor 是 torch.Tensor 类型
                if isinstance(tensor, torch.Tensor):
                    tensor = tensor.to(device)
                new_it[key] = tensor
            intermediate_tensors = IntermediateTensors(tensors=new_it)
            del broadcast_data['intermediate_tensors']

        worker_input = WorkerInput.from_broadcasted_tensor_dict(broadcast_data)
        model_input = (
            self.model_runner.make_model_input_from_broadcasted_tensor_dict(
                broadcast_data))

        kwargs = extract_previous_hidden_states(broadcast_data)

        return model_input, worker_input, kwargs, intermediate_tensors


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