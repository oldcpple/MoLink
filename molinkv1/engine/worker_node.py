"""
MoLink Worker Node — lightweight node that directly owns a model runner.

Unlike the head node (which uses MultiprocExecutor with separate worker processes),
the worker node initializes a single MolinkWorker in-process and calls its model
runner directly from the gRPC handler.  No shared-memory message queues, no
collective_rpc.
"""

import asyncio
import pickle
import struct
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional

import cloudpickle
import grpc.aio as aio
import numpy as np
import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.sequence import IntermediateTensors
from vllm.v1.outputs import ModelRunnerOutput

from molinkv1.comm import molink_pb2, molink_pb2_grpc
from molinkv1.parallel_state import (
    init_molink_parallel_state,
    apply_molink_patches,
    is_molink_last_stage,
)
from molinkv1.utils import (
    extract_ip,
    find_free_port,
    get_grpc_options,
    PipelineTopology,
    serialize_metadata,
    deserialize_metadata,
)
from molinkv1.worker.worker import MolinkWorker

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Tensor serialization helpers (same wire format as executor.py)
# ---------------------------------------------------------------------------

def _serialize_tensors(tensors_cpu: dict[str, torch.Tensor]) -> molink_pb2.IntermediateTensors:
    grpc_tensors = molink_pb2.IntermediateTensors()
    for key, tensor in tensors_cpu.items():
        tensor = tensor.detach().cpu()
        shape = tensor.shape
        torch_dtype = str(tensor.dtype)
        if tensor.dtype == torch.bfloat16:
            t_bf = tensor.contiguous()
            if t_bf.dim() == 0:
                raw = t_bf.unsqueeze(0).view(dtype=torch.uint8).numpy().tobytes()
            else:
                raw = t_bf.view(dtype=torch.uint8).numpy().tobytes()
        else:
            raw = tensor.numpy().tobytes()
        header = struct.pack("<I", len(shape))
        for dim in shape:
            header += struct.pack("<Q", dim)
        dtype_bytes = torch_dtype.encode("ascii")
        header += struct.pack("<I", len(dtype_bytes)) + dtype_bytes
        grpc_tensors.tensors.append(
            molink_pb2.TensorEntry(key=key, tensor_data=header + raw)
        )
    return grpc_tensors


def _deserialize_tensors(tensor_bytes: Dict[str, bytes]) -> IntermediateTensors:
    tensors = {}
    for key, data in tensor_bytes.items():
        offset = 0
        (ndim,) = struct.unpack_from("<I", data, offset); offset += 4
        shape = []
        for _ in range(ndim):
            (dim,) = struct.unpack_from("<Q", data, offset); shape.append(dim); offset += 8
        (dtype_len,) = struct.unpack_from("<I", data, offset); offset += 4
        dtype_name = data[offset : offset + dtype_len].decode("ascii"); offset += dtype_len
        raw = data[offset:]
        if dtype_name == "torch.bfloat16":
            n_elements = 1
            for d in shape:
                n_elements *= d
            tensor = (
                torch.frombuffer(raw, dtype=torch.uint8)
                .reshape(n_elements, 2)
                .view(torch.bfloat16)
                .reshape(tuple(shape))
                .to("cuda")
            )
        else:
            np_array = np.frombuffer(
                raw, dtype=np.dtype(dtype_name.replace("torch.", ""))
            ).reshape(tuple(shape))
            tensor = torch.from_numpy(np_array).to("cuda")
        tensors[key] = tensor
    return IntermediateTensors(tensors=tensors)


# ---------------------------------------------------------------------------
# gRPC Service for the worker node
# ---------------------------------------------------------------------------

class WorkerNodeService(molink_pb2_grpc.MolinkServiceServicer):
    """gRPC service that drives the local model runner directly."""

    def __init__(
        self,
        worker: MolinkWorker,
        executor_pool: ThreadPoolExecutor,
        head_ip: str,
        start_layer: int,
        end_layer: int,
        max_message_size_mb: int = 200,
    ):
        self.worker = worker
        self._pool = executor_pool
        self.max_message_size_mb = max_message_size_mb
        self.topology = PipelineTopology(head_ip, start_layer, end_layer)

        # Stub cache for sending results back to other nodes.
        self._stub_cache: Dict[str, molink_pb2_grpc.MolinkServiceStub] = {}
        self._channel_cache: Dict[str, aio.Channel] = {}

        # Thread-safe metrics
        self._metrics_lock = threading.Lock()
        self._metrics_deque = collections.deque(maxlen=2000)

    def _get_stub(self, address: str) -> molink_pb2_grpc.MolinkServiceStub:
        if address not in self._stub_cache:
            channel = aio.insecure_channel(address, options=get_grpc_options(self.max_message_size_mb))
            self._channel_cache[address] = channel
            self._stub_cache[address] = molink_pb2_grpc.MolinkServiceStub(channel)
        return self._stub_cache[address]

    # -- topology -----------------------------------------------------------

    async def JoinPipeline(self, request, context):
        self.topology.add_node(request.ip, request.start_layer, request.end_layer)
        logger.info(f"Node {request.ip} joined pipeline (layers {request.start_layer}-{request.end_layer})")
        return molink_pb2.GrpcResponseData(res=1)

    async def GetTopology(self, request, context):
        nodes = [molink_pb2.NodeInfo(ip=n["ip"], start_layer=n["start_layer"], end_layer=n["end_layer"])
                 for n in self.topology.node_pool]
        return molink_pb2.PipelineTopology(nodes=nodes)

    # -- data handlers ------------------------------------------------------

    async def PushIntermediateTensors(self, request, context):
        virtual_engine = request.virtual_engine
        intermediate_tensors_bytes = {}
        for entry in request.intermediate_tensors.tensors:
            intermediate_tensors_bytes[entry.key] = entry.tensor_data
        grpc_metadata = deserialize_metadata(request.grpc_metadata)
        scheduler_output_bytes = request.scheduler_output

        loop = asyncio.get_running_loop()
        intermediate_tensors = await loop.run_in_executor(
            None, _deserialize_tensors, intermediate_tensors_bytes
        )
        scheduler_output = cloudpickle.loads(scheduler_output_bytes)

        # Execute the full forward + sample step directly on the local model runner.
        output = await self._run_step(scheduler_output, intermediate_tensors)

        # Route the result.
        server_list = grpc_metadata.get("server_list", [])
        my_address = f"{self._ip}:{self._grpc_port}"
        try:
            my_idx = server_list.index(my_address)
        except ValueError:
            my_idx = len(server_list) - 1

        is_last_stage = (my_idx == len(server_list) - 1)

        if is_last_stage:
            # Send result back to head node.
            head_server = grpc_metadata.get("head")
            output_bytes = cloudpickle.dumps(output, protocol=pickle.HIGHEST_PROTOCOL)
            await self._push_sampler_output(output_bytes, virtual_engine, head_server)
        else:
            # Send intermediate tensors to next stage.
            next_server = server_list[my_idx + 1]
            tensors = output.tensors if isinstance(output, IntermediateTensors) else {"hidden_states": output}
            await self._push_intermediate_tensors(
                tensors, scheduler_output_bytes, grpc_metadata, virtual_engine, next_server
            )

        return molink_pb2.GrpcResponseData(res=1)

    async def PushSamplerOutput(self, request, context):
        # Worker nodes don't normally receive sampler output, but keep for compatibility.
        return molink_pb2.GrpcResponseData(res=1)

    async def ExecuteWorkerStep(self, request, context):
        # Not used in the new architecture — data arrives via PushIntermediateTensors.
        return molink_pb2.GrpcResponseData(res=1)

    async def HealthCheck(self, request, context):
        return molink_pb2.HealthCheckResponse(status="healthy")

    # -- internal -----------------------------------------------------------

    async def _run_step(self, scheduler_output, intermediate_tensors):
        """Run execute_model + sample_tokens on the local worker."""
        loop = asyncio.get_running_loop()

        # Set intermediate tensors.
        self.worker._molink_set_intermediate_tensors(intermediate_tensors)

        # Execute model forward pass.
        output = await loop.run_in_executor(
            None, self.worker.execute_model, scheduler_output
        )

        # If output is None (last PP stage stores state), call sample_tokens.
        if output is None:
            output = await loop.run_in_executor(
                None, self.worker.sample_tokens, None
            )

        return output

    async def _push_sampler_output(self, output_bytes, virtual_engine, head_server):
        request = molink_pb2.SamplerOutput(
            output_data=output_bytes, virtual_engine=virtual_engine
        )
        stub = self._get_stub(head_server)
        await stub.PushSamplerOutput(request)

    async def _push_intermediate_tensors(self, tensors, scheduler_output_bytes,
                                          grpc_metadata, virtual_engine, next_server):
        tensors_cpu = {k: v.to("cpu") for k, v in tensors.items()}
        grpc_tensors = _serialize_tensors(tensors_cpu)
        request = molink_pb2.GrpcRequestData(
            scheduler_output=scheduler_output_bytes,
            intermediate_tensors=grpc_tensors,
            grpc_metadata=serialize_metadata(grpc_metadata),
            virtual_engine=virtual_engine,
        )
        stub = self._get_stub(next_server)
        await stub.PushIntermediateTensors(request)


import collections  # noqa: E402 — needed for deque


# ---------------------------------------------------------------------------
# Worker Node — top-level orchestrator
# ---------------------------------------------------------------------------

class MolinkWorkerNode:
    """Lightweight worker node that directly owns a model runner.

    Initialisation sequence:
    1. Create MolinkWorker (in-process, no separate worker process)
    2. init_device → load_model → determine_available_memory →
       initialize_from_config → compile_or_warm_up_model
    3. Start gRPC server
    4. Join pipeline (connect to head node)
    """

    def __init__(self, vllm_config: VllmConfig):
        from molinkv1.config import MolinkConfig
        self.molink_config: MolinkConfig = getattr(vllm_config, "molink_config", None)
        self.vllm_config = vllm_config
        self.grpc_server: Optional[aio.Server] = None
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_thread: Optional[threading.Thread] = None
        self._pool = ThreadPoolExecutor(max_workers=10)

        # Initialize MoLink parallel state.
        start_layer, end_layer = self.molink_config.get_serving_layers()
        init_molink_parallel_state(
            enabled=True, start_layer=start_layer, end_layer=end_layer,
        )

        # ---- Create worker directly (no MultiprocExecutor) ----
        dist_port = find_free_port(start_port=29500)
        self.worker = MolinkWorker(
            vllm_config=vllm_config,
            local_rank=0,
            rank=0,
            distributed_init_method=f"tcp://127.0.0.1:{dist_port}",
            is_driver_worker=True,
        )

        # Initialize CUDA, distributed, model runner, KV cache, and warm up.
        from vllm.config import set_current_vllm_config
        with set_current_vllm_config(vllm_config):
            self.worker.init_device()
            self.worker.load_model()
            self._init_kv_cache()
            self.worker.compile_or_warm_up_model()

        logger.info("MolinkWorkerNode: model runner ready")

        # ---- Start gRPC server ----
        self.ip = extract_ip()
        self.grpc_port = find_free_port(
            start_port=self.molink_config.grpc_port if self.molink_config.grpc_port > 0 else 50051
        )
        self.grpc_address = f"{self.ip}:{self.grpc_port}"

        self._start_event_loop_thread()
        future = asyncio.run_coroutine_threadsafe(
            self._start_grpc_server(start_layer, end_layer), self._event_loop
        )
        future.result(timeout=30)

        # ---- Join pipeline ----
        future = asyncio.run_coroutine_threadsafe(
            self._join_pipeline(), self._event_loop
        )
        try:
            future.result(timeout=30)
        except Exception as e:
            logger.error(f"Failed to join pipeline: {e}")

        logger.info(
            f"MolinkWorkerNode initialized at {self.grpc_address}, "
            f"layers {start_layer}-{end_layer}"
        )

    # -- KV cache -----------------------------------------------------------

    def _init_kv_cache(self):
        """Profile memory, compute KV cache config, allocate and initialize."""
        from vllm.v1.core.kv_cache_utils import (
            get_kv_cache_configs,
            generate_scheduler_kv_cache_config,
        )

        available_memory = self.worker.determine_available_memory()
        if isinstance(available_memory, list):
            available_memory = available_memory[0]

        kv_cache_specs = self.worker.get_kv_cache_spec()
        if isinstance(kv_cache_specs, list):
            kv_cache_specs = kv_cache_specs[0]

        kv_cache_configs = get_kv_cache_configs(
            self.vllm_config, [kv_cache_specs], [available_memory]
        )

        scheduler_kv_cache_config = generate_scheduler_kv_cache_config(kv_cache_configs)
        self.vllm_config.cache_config.num_gpu_blocks = scheduler_kv_cache_config.num_blocks
        kv_cache_groups = scheduler_kv_cache_config.kv_cache_groups
        if kv_cache_groups:
            self.vllm_config.cache_config.block_size = min(
                g.kv_cache_spec.block_size for g in kv_cache_groups
            )

        self.worker.initialize_from_config(kv_cache_configs[0])

        logger.info(
            f"KV cache initialized: num_gpu_blocks={scheduler_kv_cache_config.num_blocks}"
        )

    # -- Event loop ---------------------------------------------------------

    def _start_event_loop_thread(self):
        loop_ready = threading.Event()

        def run_loop():
            self._event_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._event_loop)
            loop_ready.set()
            self._event_loop.run_forever()

        self._loop_thread = threading.Thread(target=run_loop, daemon=True, name="MolinkWorkerEventLoop")
        self._loop_thread.start()
        loop_ready.wait(timeout=10)

    async def _start_grpc_server(self, start_layer, end_layer):
        config = self.molink_config
        self.grpc_server = aio.server(self._pool, options=get_grpc_options(config.max_message_size_mb))

        service = WorkerNodeService(
            worker=self.worker,
            executor_pool=self._pool,
            head_ip=f"{self.ip}:{self.grpc_port}",
            start_layer=start_layer,
            end_layer=end_layer,
        )
        service._ip = self.ip
        service._grpc_port = self.grpc_port
        self.service = service

        molink_pb2_grpc.add_MolinkServiceServicer_to_server(service, self.grpc_server)
        self.grpc_server.add_insecure_port(f"[::]:{self.grpc_port}")
        await self.grpc_server.start()
        logger.info(f"Worker gRPC server started on port {self.grpc_port}")

    async def _join_pipeline(self):
        config = self.molink_config
        channel = None
        try:
            channel = aio.insecure_channel(
                config.initial_peer,
                options=get_grpc_options(config.max_message_size_mb),
            )
            stub = molink_pb2_grpc.MolinkServiceStub(channel)
            start_layer, end_layer = config.get_serving_layers()
            node_info = molink_pb2.NodeInfo(
                ip=self.grpc_address, start_layer=start_layer, end_layer=end_layer
            )
            response = await stub.JoinPipeline(node_info)
            if response.res == 1:
                logger.info(f"Successfully joined pipeline at {config.initial_peer}")
            else:
                logger.error(f"Failed to join pipeline: {response.error_message}")
        except Exception as e:
            logger.error(f"Error joining pipeline: {e}")
            traceback.print_exc()
            raise
        finally:
            if channel is not None:
                await channel.close()

    # -- Lifecycle ----------------------------------------------------------

    def shutdown(self):
        from molinkv1.parallel_state import destroy_molink_parallel_state
        destroy_molink_parallel_state()

        if self._event_loop and self._event_loop.is_running():
            future = asyncio.run_coroutine_threadsafe(self._async_shutdown(), self._event_loop)
            try:
                future.result(timeout=10)
            except Exception:
                pass
            self._event_loop.call_soon_threadsafe(self._event_loop.stop)

        if self._loop_thread and self._loop_thread.is_alive():
            self._loop_thread.join(timeout=5)

        self._pool.shutdown(wait=False)
        logger.info("MolinkWorkerNode shutdown complete")

    async def _async_shutdown(self):
        if self.grpc_server:
            await self.grpc_server.stop(grace=5)
        for channel in self.service._channel_cache.values():
            await channel.close()

    # -- API surface for api_server.py compat --------------------------------

    def get_communication_metrics(self) -> dict:
        return {"node": self.grpc_address, "is_head": False}

    def reset_communication_metrics(self):
        pass
