import asyncio
import grpc
import time
import torch
import json
import io
import msgspec
from concurrent import futures
from typing import Any, Awaitable, Dict, List, Optional, Set, Tuple, Union
from vllm.executor.mp_distributed_executor import MultiprocessingDistributedExecutor
from vllm.executor.multiproc_worker_utils import (
    ProcessWorkerWrapper, ResultHandler, WorkerMonitor,
    set_multiprocessing_worker_envs)
from vllm.utils import (_run_task_with_lock, get_distributed_init_method,
                        get_ip, get_open_port, make_async)
from vllm.config import VllmConfig
from vllm.sequence import ExecuteModelRequest, SamplerOutput, IntermediateTensors
from vllm.distributed import get_pp_group
from molink.worker.worker_base import MolinkWorkerWrapperBase
from molink.config import MolinkConfig
from molink.dht.proto import comm_pb2, comm_pb2_grpc
from molink.dht.comm_handler import CommService
from molink.dht.dht import DHTNode
from molink.dht.pipeline_manager import PipelineManager


class MolinkMultiprocessingDistributedExecutor(MultiprocessingDistributedExecutor):
    
    def __init__(
        self,
        vllm_config: VllmConfig,
    ) -> None:
        self.vllm_config = vllm_config
        self.pipeline_config = vllm_config.pipeline_config
        self.parallel_config = vllm_config.parallel_config
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

        self.parallel_worker_tasks: Optional[Union[Any, Awaitable[Any]]] = None
        self.dht_node = None
        self.pipeline_manager = None
        self.comm_handler = None
        self.grpc_server = None

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

        initial_peer = self.pipeline_config.initial_peer
        model_name = self.vllm_config.model_config.model
        start_layer = self.pipeline_config.serving_layers[0]
        end_layer = self.pipeline_config.serving_layers[1]
        self.dht_node = DHTNode(initial_peer, model_name, start_layer, end_layer)
        self.pipeline_manager = PipelineManager(self.dht_node)
        self.comm_handler = CommService(self.pipeline_config.pipeline_parallel_size, self.driver_worker)
        self.grpc_server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        comm_pb2_grpc.add_ExampleServiceServicer_to_server(CommService(), self.grpc_server)
        port = self.dht_node.node_info.get('grpc_port')
        self.grpc_server.add_insecure_port('[::]:{}'.format(port))
        self.grpc_server.start()
        self.grpc_server.wait_for_termination()


    async def _driver_execute_model_async(
        self,
        execute_model_req: Optional[ExecuteModelRequest] = None,
    ) -> List[SamplerOutput]:
        if self.pp_locks is None:
            # This locks each pipeline parallel stage so multiple virtual
            # engines can't execute on the same stage at the same time
            # We create the locks here to avoid creating them in the constructor
            # which uses a different asyncio loop.
            self.pp_locks = [
                asyncio.Lock()
                for _ in range(self.parallel_config.pipeline_parallel_size)
            ]

        grpc_metadata = self.pipeline_manager.pipeline_info
        server_list = grpc_metadata.get('server_list')

        tasks = [
            asyncio.create_task(
                _run_task_with_lock(self.executing_head_server, self.pp_locks[0],
                                    execute_model_req, grpc_metadata))
        ]

        stub_list = [comm_pb2_grpc.CommServiceStub(grpc.insecure_channel(server)) for server in server_list]
        virtual_engine = execute_model_req.virtual_engine

        for pp_rank, stub in enumerate(stub_list,
                                                start=1):
            tasks.append(
                asyncio.create_task(
                    _run_task_with_lock(stub.executing_worker_step,
                                        self.pp_locks[pp_rank],
                                        virtual_engine)))
            
        results = await self.comm_handler.output_queue[virtual_engine].get()

        # Only the last PP stage has the final results.
        return results
    
    def execute_model(
        self,
        execute_model_req: Optional[ExecuteModelRequest] = None,
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> Optional[List[SamplerOutput]]:
        """Executes at least one model step on the given sequences, unless no
        sequences are provided."""
        start_time = time.perf_counter()

        inputs = self.prepare_input(execute_model_req)
        if inputs is None:
            return None

        model_input, worker_input, kwargs = inputs
        num_steps = worker_input.num_steps
        if (execute_model_req is not None and execute_model_req.spec_step_idx):
            kwargs["spec_step_idx"] = execute_model_req.spec_step_idx

        self.execute_worker(worker_input)

        # If there is no input, we don't need to execute the model.
        if worker_input.num_seq_groups == 0:
            return []

        orig_model_execute_time = 0.0
        if not get_pp_group().is_first_rank:
            intermediate_tensors = None
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

        if (self.observability_config is not None
                and self.observability_config.collect_model_execute_time
                and output is not None):
            for o in output:
                o.model_execute_time = (orig_model_execute_time +
                                        model_execute_time)

        # output is List[SamplerOutput]
        return output
    
    async def executing_head_server(self, execute_model_req: Optional[ExecuteModelRequest] = None, grpc_metadata: Optional[dict] = None):

        virtual_engine = execute_model_req.virtual_engine

        # if pipeline parallel size >= 2, it's IntermediateTensors
        # otherwise it's [SamplerOutput]
        outputs = await self.execute_model(execute_model_req)

        server_list = grpc_metadata.get('server_list')
        if len(server_list) <= 0:
            self.comm_handler.output_queue[virtual_engine].put_nowait(outputs)
            return
        # ip : grpc_port
        next_server = server_list[0]
        server_list = server_list[1:]
        grpc_metadata.update({'server_list' : server_list})
        grpc_metadata = json.dumps(grpc_metadata).encode('utf-8')
        execute_model_req.async_callback = None
        bytes_emr = msgspec.json.encode(execute_model_req)

        intermediate_tensors = outputs
        grpc_intermediate_tensors = comm_pb2.IntermediateTensors()
        for key, tensors in intermediate_tensors.items():
            buffer = io.BytesIO()
            torch.save(tensors, buffer)
            byte_data = buffer.getvalue()
            grpc_intermediate_tensors.tensors.append(comm_pb2.TensorEntry(key = key,
                                                                                tensor_data = byte_data))

        grpc_request_data = comm_pb2.GrpcRequestData(execute_model_request = bytes_emr,
                                                        intermediate_tensors = grpc_intermediate_tensors,
                                                        grpc_metadata = grpc_metadata,
                                                        virtual_engine = virtual_engine)
        
        with grpc.insecure_channel(next_server) as channel:
            stub = comm_pb2_grpc.CommServiceStub(channel)
            res = await stub.PushIntermediateTensors(grpc_request_data)