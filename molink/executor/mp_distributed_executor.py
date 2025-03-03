import asyncio
import time
import torch
import json
import io
import msgspec
import grpc
import copy
import traceback
from grpc import aio
from concurrent import futures
from typing import Any, Awaitable, Dict, List, Optional, Set, Tuple, Union
from vllm.executor.mp_distributed_executor import MultiprocessingDistributedExecutor
from vllm.executor.multiproc_worker_utils import (
    ProcessWorkerWrapper, ResultHandler, WorkerMonitor,
    set_multiprocessing_worker_envs)
from vllm.utils import (_run_task_with_lock, get_distributed_init_method,
                        get_ip, get_open_port, make_async)
from vllm.config import VllmConfig
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.distributed import get_pp_group
from vllm.sequence import IntermediateTensors, ExecuteModelRequest
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
        self.preset_next_server = None
        self.channel_to_next_server = None

        self._init_executor()

    def _init_executor(self) -> None:
        # Create the parallel GPU workers.

        # for testing
        world_size = self.parallel_config.tensor_parallel_size
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
        self.comm_handler = CommService(self.parallel_config.pipeline_parallel_size, self)
        self.grpc_server = aio.server(futures.ThreadPoolExecutor(max_workers=10),
                                            options=[
                                                ('grpc.max_send_message_length', 200 * 1024 * 1024),  # 200MB
                                                ('grpc.max_receive_message_length', 200 * 1024 * 1024)  # 200MB
                                            ])
        comm_pb2_grpc.add_CommServiceServicer_to_server(self.comm_handler, self.grpc_server)
        port = self.dht_node.node_info.grpc_port
        self.grpc_server.add_insecure_port('[::]:{}'.format(port))
        asyncio.create_task(self._start_grpc_server())

        grpc_info = f'{self.dht_node.ip}:{self.dht_node.node_info.grpc_port}'
        dht_info = f'{self.dht_node.ip}:{self.dht_node.node_info.dht_port}'

        print("DISTRIBUTED SERVICE INFO: MoLink gRPC server works at {}, ".format(grpc_info))
        print("DISTRIBUTED SERVICE INFO: MoLink DHT server works at {}".format(dht_info))
        print("DISTRIBUTED SERVICE INFO: If this is the first node of the swarm, you can copy the DHT INFO as the initial peer of following nodes")



    async def _start_grpc_server(self):
        try:

            await self.grpc_server.start()
            await self.grpc_server.wait_for_termination()

        except asyncio.CancelledError:
            await self.grpc_server.stop(grace=5)

    def _establish_conn_with_next_server(self, next_server):
        # will be trigger during the ever first run
        try:

            if self.channel_to_next_server is not None:
                del self.channel_to_next_server
            self.channel_to_next_server = aio.insecure_channel(next_server,
                                    options=[
                                        ('grpc.max_send_message_length', 200 * 1024 * 1024),  # 200MB
                                        ('grpc.max_receive_message_length', 200 * 1024 * 1024)  # 200MB
                                    ])

        except Exception as e:
            print('Encounter the following exception: {}'.format(e))
            traceback.print_exc()


    async def execute_model_async(
            self,
            execute_model_req: ExecuteModelRequest) -> List[SamplerOutput]:
        try:

            if self.parallel_worker_tasks is None:
                # Start model execution loop running in the parallel workers
                self.parallel_worker_tasks = asyncio.create_task(
                    self._start_worker_execution_loop())
                
        except Exception as e:
            print('Encounter the following exception: {}'.format(e))
            traceback.print_exc()

        # Only the driver worker returns the sampling results.
        return await self._driver_execute_model_async(execute_model_req)

    async def _driver_execute_model_async(
        self,
        execute_model_req: Optional[ExecuteModelRequest] = None,
    ) -> List[SamplerOutput]:
        try:
            # bypass the empty execution
            if execute_model_req is None:
                return

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
            if len(grpc_metadata) <= 0:
                server_list_raw = []
            else:
                server_list_raw = grpc_metadata.get('server_list')
            # pop the head server
            server_list = server_list_raw[1:]

            tasks = [
                asyncio.create_task(
                    _run_task_with_lock(self.executing_head_server, self.pp_locks[0],
                                        execute_model_req, grpc_metadata))
            ]

            stub_list = [comm_pb2_grpc.CommServiceStub(aio.insecure_channel(server)) for server in server_list]
            virtual_engine = execute_model_req.virtual_engine

            trigger_request = comm_pb2.GrpcTriggerRequest(virtual_engine = virtual_engine)

            for pp_rank, stub in enumerate(stub_list,
                                                    start=1):
                tasks.append(
                    asyncio.create_task(
                        _run_task_with_lock(stub.ExecutingWorkerStep,
                                            self.pp_locks[pp_rank],
                                            trigger_request)))
                
            results = await self.comm_handler.output_queue[virtual_engine].get()

            return results
        
        except Exception as e:
            print('Encounter the following exception: {}'.format(e))
            traceback.print_exc()
    
    async def executing_head_server(self, execute_model_req: Optional[ExecuteModelRequest] = None, grpc_metadata: Optional[dict] = None):

        try:
            virtual_engine = execute_model_req.virtual_engine

            # if pipeline parallel size >= 2, it's [IntermediateTensors.tensors]
            # otherwise it's [SamplerOutput]
            outputs = await self.driver_exec_model(execute_model_req)

            if len(grpc_metadata) <= 0:
                server_list = []
            else:
                server_list = grpc_metadata.get('server_list')
            # ip : grpc_port

            if len(server_list) <= 1:
                self.comm_handler.output_queue[virtual_engine].put_nowait(outputs)
                return
            
            outputs = outputs[0]
            next_server = server_list[1]
            grpc_metadata = json.dumps(grpc_metadata).encode('utf-8')
            execute_model_req.async_callback = None

            len_seq_group = len(execute_model_req.seq_group_metadata_list)
            for i in range(len_seq_group):
                seq_data_dict = execute_model_req.seq_group_metadata_list[i].seq_data
                for idx, seq_data in seq_data_dict.items():
                    seq_data._prompt_token_ids = list(seq_data._prompt_token_ids)
                    seq_data._output_token_ids = list(seq_data._output_token_ids)
                    seq_data_dict.update({idx : seq_data})

                execute_model_req.seq_group_metadata_list[i].seq_data = seq_data_dict

            #execute_model_req.seq_group_metadata_list = None
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
            
            # case 1: first run
            # case 2: the pipeline path has changed
            if self.preset_next_server is None or self.preset_next_server != next_server:
                self.preset_next_server = next_server
                self._establish_conn_with_next_server(next_server)

            assert self.channel_to_next_server is not None, 'Connection to next server has not been properly set'
            stub = comm_pb2_grpc.CommServiceStub(self.channel_to_next_server)
            res = await stub.PushIntermediateTensors(grpc_request_data)

        except Exception as e:
            print('Encounter the following exception: {}'.format(e))
            traceback.print_exc()