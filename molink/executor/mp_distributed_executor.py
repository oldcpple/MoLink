import asyncio
from typing import Any, List, Optional
from vllm.executor.mp_distributed_executor import MultiprocessingDistributedExecutor
from vllm.executor.multiproc_worker_utils import (
    ProcessWorkerWrapper, ResultHandler, WorkerMonitor,
    set_multiprocessing_worker_envs)
from vllm.utils import (_run_task_with_lock, get_distributed_init_method,
                        get_ip, get_open_port, make_async)
from molink.worker.worker_base import MolinkWorkerWrapperBase
from molink.config import MolinkConfig

class MolinkMultiprocessingDistributedExecutor(MultiprocessingDistributedExecutor):
    
    def __init__(
        self,
        vllm_config: VllmConfig,
    ) -> None:
        self.vllm_config = vllm_config
        self.pipeline_config = vllm_config.pipeline_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.load_config = vllm_config.load_config
        self.parallel_config = vllm_config.parallel_config
        self.scheduler_config = vllm_config.scheduler_config
        self.device_config = vllm_config.device_config
        self.speculative_config = vllm_config.speculative_config
        self.prompt_adapter_config = vllm_config.prompt_adapter_config
        self.observability_config = vllm_config.observability_config
        self._init_executor()

    def _init_executor(self) -> None:
        # Create the parallel GPU workers.
        world_size = self.parallel_config.world_size
        tensor_parallel_size = self.parallel_config.tensor_parallel_size

        # Set multiprocessing envs that are common to V0 and V1
        set_multiprocessing_worker_envs(self.parallel_config)

        # Multiprocessing-based executor does not support multi-node setting.
        # Since it only works for single node, we can use the loopback address
        # 127.0.0.1 for communication.
        distributed_init_method = get_distributed_init_method(
            "127.0.0.1", get_open_port())

        self.workers: List[ProcessWorkerWrapper] = []
        # This is the list of workers that are rank 0 of each TP group EXCEPT
        # global rank 0. These are the workers that will broadcast to the
        # rest of the workers.
        self.tp_driver_workers: List[ProcessWorkerWrapper] = []
        # This is the list of workers that are not drivers and not the first
        # worker in a TP group. These are the workers that will be
        # broadcasted to.
        self.non_driver_workers: List[ProcessWorkerWrapper] = []

        if world_size == 1:
            self.worker_monitor = None
        else:
            result_handler = ResultHandler()
            for rank in range(1, world_size):
                worker = ProcessWorkerWrapper(result_handler,
                                              MolinkWorkerWrapperBase,
                                              self.vllm_config, rank)
                self.workers.append(worker)
                if rank % tensor_parallel_size == 0:
                    self.tp_driver_workers.append(worker)
                else:
                    self.non_driver_workers.append(worker)

            self.worker_monitor = WorkerMonitor(self.workers, result_handler)
            result_handler.start()
            self.worker_monitor.start()

        # Set up signal handlers to shutdown the executor cleanly
        # sometimes gc does not work well

        self.driver_worker = MolinkWorkerWrapperBase(self.vllm_config, 0)

        all_kwargs = []
        distributed_init_method = get_distributed_init_method(
            get_ip(), get_open_port())
        for i in range(world_size):
            local_rank = i
            rank = i
            kwargs = dict(
                vllm_config=self.vllm_config,
                local_rank=local_rank,
                rank=rank,
                distributed_init_method=distributed_init_method,
                is_driver_worker=(not self.parallel_config)
                or (rank % self.parallel_config.tensor_parallel_size == 0),
            )
            all_kwargs.append(kwargs)
        
        _is_first_rank = self.pipeline_config._is_first_rank
        _is_last_rank = self.pipeline_config._is_last_rank
        self._run_workers("init_worker", all_kwargs)
        self._run_workers("init_device", _is_first_rank, _is_last_rank)
        self._run_workers("load_model",
                          max_concurrent_workers=self.parallel_config.
                          max_parallel_loading_workers)
        self.driver_exec_model = make_async(self.driver_worker.execute_model)
        self.pp_locks: Optional[List[asyncio.Lock]] = None