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
                torch.frombuffer(bytearray(raw), dtype=torch.uint8)
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

        # Serialize pipeline-step execution so only one _run_step runs at a time.
        self._step_lock = asyncio.Lock()

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

        # DEBUG: log received tensors
        import sys
        for key, tensor in intermediate_tensors.tensors.items():
            print(f"[MOLINK-DEBUG][WORKER] Received tensor '{key}': shape={tensor.shape}, dtype={tensor.dtype}, "
                  f"device={tensor.device}, first5={tensor.flatten()[:5].tolist()}",
                  file=sys.stderr, flush=True)

        scheduler_output = cloudpickle.loads(scheduler_output_bytes)

        # Serialize pipeline-step execution to prevent concurrent access
        # to the shared model runner state.
        async with self._step_lock:
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

        import sys
        print(f"[MOLINK-DEBUG][WORKER] execute_model returned: type={type(output).__name__}",
              file=sys.stderr, flush=True)

        # DEBUG: check logits BEFORE sample_tokens clears them
        if output is None:
            try:
                mr = self.worker.model_runner
                state = getattr(mr, 'execute_model_state', None)
                if state is not None:
                    logits = state[1]  # logits at index 1 in the tuple
                    if logits is not None:
                        probs = torch.softmax(logits.float(), dim=-1)
                        top5_vals, top5_ids = torch.topk(probs[0], 5)
                        print(f"[MOLINK-DEBUG][WORKER-LOGITS] shape={logits.shape}, "
                              f"top5_ids={top5_ids.tolist()}, top5_probs={top5_vals.tolist()}",
                              file=sys.stderr, flush=True)
                    else:
                        print(f"[MOLINK-DEBUG][WORKER-LOGITS] logits is None",
                              file=sys.stderr, flush=True)
                else:
                    print(f"[MOLINK-DEBUG][WORKER-LOGITS] execute_model_state is None",
                          file=sys.stderr, flush=True)
            except Exception as e:
                import traceback as _tb
                print(f"[MOLINK-DEBUG][WORKER-LOGITS] Error: {e}\n{_tb.format_exc()}",
                      file=sys.stderr, flush=True)

        # If output is None (last PP stage stores state), call sample_tokens.
        if output is None:
            output = await loop.run_in_executor(
                None, self.worker.sample_tokens, None
            )
            print(f"[MOLINK-DEBUG][WORKER] sample_tokens returned: type={type(output).__name__}",
                  file=sys.stderr, flush=True)

        # Resolve async output (contains unpicklable torch.Event/Stream).
        from vllm.v1.outputs import AsyncModelRunnerOutput
        if isinstance(output, AsyncModelRunnerOutput):
            output = await loop.run_in_executor(None, output.get_output)
            print(f"[MOLINK-DEBUG][WORKER] Resolved AsyncModelRunnerOutput: type={type(output).__name__}",
                  file=sys.stderr, flush=True)

        # DEBUG: log output details
        if hasattr(output, 'sampled_token_ids'):
            print(f"[MOLINK-DEBUG][WORKER] sampled_token_ids={output.sampled_token_ids}",
                  file=sys.stderr, flush=True)
        if hasattr(output, 'req_ids'):
            print(f"[MOLINK-DEBUG][WORKER] req_ids={output.req_ids}",
                  file=sys.stderr, flush=True)

        # DEBUG: check logits from model runner
        try:
            state = self.worker.model_runner.execute_model_state
            if state is not None:
                logits = state[1]  # logits is at index 1
                if logits is not None:
                    probs = torch.softmax(logits.float(), dim=-1)
                    top5_vals, top5_ids = torch.topk(probs[0], 5)
                    print(f"[MOLINK-DEBUG][WORKER] Logits shape={logits.shape}, "
                          f"top5_ids={top5_ids.tolist()}, top5_probs={top5_vals.tolist()}",
                          file=sys.stderr, flush=True)
        except Exception as e:
            print(f"[MOLINK-DEBUG][WORKER] Failed to check logits: {e}",
                  file=sys.stderr, flush=True)

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

        # Initialize CUDA and load model.
        from vllm.config import set_current_vllm_config
        with set_current_vllm_config(vllm_config):
            self.worker.init_device()
            self.worker.load_model()

        # DEBUG: Check which layers the model has
        import sys
        mr = self.worker.model_runner
        model = mr.model

        # Check PP config from model
        pp_size = vllm_config.parallel_config.pipeline_parallel_size
        print(f"[MOLINK-DEBUG][WORKER-MODEL] pp_size={pp_size}",
              file=sys.stderr, flush=True)

        # Check the actual model layer range
        inner_model = getattr(model, 'model', model)
        if hasattr(inner_model, 'layers'):
            num_layers = len(inner_model.layers)
            # Check if model has start_layer/end_layer stored
            start = getattr(inner_model, 'start_layer', 'N/A')
            end = getattr(inner_model, 'end_layer', 'N/A')
            pp_start = getattr(inner_model, 'pp_start_layer', 'N/A')
            pp_end = getattr(inner_model, 'pp_end_layer', 'N/A')
            print(f"[MOLINK-DEBUG][WORKER-MODEL] num_layers={num_layers}, "
                  f"start_layer={start}, end_layer={end}, "
                  f"pp_start_layer={pp_start}, pp_end_layer={pp_end}",
                  file=sys.stderr, flush=True)
            # Check first and last layer weight norm
            first_layer = inner_model.layers[0]
            last_layer = inner_model.layers[-1]
            for name, param in list(first_layer.named_parameters())[:1]:
                print(f"[MOLINK-DEBUG][WORKER-MODEL] Layer 0 '{name}': norm={param.data.norm().item():.4f}, "
                      f"is_meta={param.is_meta}",
                      file=sys.stderr, flush=True)
            for name, param in list(last_layer.named_parameters())[:1]:
                print(f"[MOLINK-DEBUG][WORKER-MODEL] Layer {num_layers-1} '{name}': norm={param.data.norm().item():.4f}, "
                      f"is_meta={param.is_meta}",
                      file=sys.stderr, flush=True)
            # Check layers 19 and 20
            if num_layers > 20:
                layer19 = inner_model.layers[19]
                layer20 = inner_model.layers[20]
                for name, param in list(layer19.named_parameters())[:1]:
                    print(f"[MOLINK-DEBUG][WORKER-MODEL] Layer 19 '{name}': norm={param.data.norm().item():.4f}, "
                          f"is_meta={param.is_meta}",
                          file=sys.stderr, flush=True)
                for name, param in list(layer20.named_parameters())[:1]:
                    print(f"[MOLINK-DEBUG][WORKER-MODEL] Layer 20 '{name}': norm={param.data.norm().item():.4f}, "
                          f"is_meta={param.is_meta}",
                          file=sys.stderr, flush=True)

        # Also check pp_indices
        from molinkv1.parallel_state import get_molink_pp_indices, is_molink_enabled
        from vllm.distributed.utils import get_pp_indices
        print(f"[MOLINK-DEBUG][WORKER-MODEL] is_molink_enabled={is_molink_enabled()}, "
              f"get_molink_pp_indices(40,0,1)={get_molink_pp_indices(40,0,1)}, "
              f"get_pp_indices(40,0,1)={get_pp_indices(40,0,1)}",
              file=sys.stderr, flush=True)

        # ---- Start gRPC server first so we can receive head's num_gpu_blocks ----
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

        # ---- Join pipeline and get head's num_gpu_blocks ----
        head_num_gpu_blocks = None
        future = asyncio.run_coroutine_threadsafe(
            self._join_pipeline(), self._event_loop
        )
        try:
            head_num_gpu_blocks = future.result(timeout=30)
        except Exception as e:
            logger.error(f"Failed to join pipeline: {e}")

        # ---- Initialize KV cache (possibly capped to head's num_gpu_blocks) ----
        with set_current_vllm_config(vllm_config):
            self._init_kv_cache(head_num_gpu_blocks=head_num_gpu_blocks)
            self.worker.compile_or_warm_up_model()

        logger.info("MolinkWorkerNode: model runner ready")

        logger.info(
            f"MolinkWorkerNode initialized at {self.grpc_address}, "
            f"layers {start_layer}-{end_layer}"
        )

    # -- KV cache -----------------------------------------------------------

    def _init_kv_cache(self, head_num_gpu_blocks: Optional[int] = None):
        """Profile memory, compute KV cache config, allocate and initialize.

        Args:
            head_num_gpu_blocks: The head node's num_gpu_blocks.  If provided,
                the worker caps its own num_gpu_blocks to this value so that
                the head scheduler never allocates blocks the worker doesn't
                have.
        """
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
        num_blocks = scheduler_kv_cache_config.num_blocks

        # Cap to the head node's value so the scheduler's block IDs are
        # valid on this worker.
        if head_num_gpu_blocks is not None and head_num_gpu_blocks > 0:
            if num_blocks > head_num_gpu_blocks:
                logger.info(
                    f"Worker num_gpu_blocks ({num_blocks}) > head "
                    f"({head_num_gpu_blocks}). Capping to head's value."
                )
                num_blocks = head_num_gpu_blocks
                self.vllm_config.cache_config.num_gpu_blocks = num_blocks
                # Regenerate config with capped blocks.
                self.vllm_config.cache_config.num_gpu_blocks_override = num_blocks
                kv_cache_configs = get_kv_cache_configs(
                    self.vllm_config, [kv_cache_specs], [available_memory]
                )
                scheduler_kv_cache_config = generate_scheduler_kv_cache_config(
                    kv_cache_configs
                )
                num_blocks = scheduler_kv_cache_config.num_blocks
            else:
                logger.info(
                    f"Worker num_gpu_blocks ({num_blocks}) <= head "
                    f"({head_num_gpu_blocks}). No capping needed."
                )
                self.vllm_config.cache_config.num_gpu_blocks = num_blocks
        else:
            self.vllm_config.cache_config.num_gpu_blocks = num_blocks

        kv_cache_groups = scheduler_kv_cache_config.kv_cache_groups
        if kv_cache_groups:
            self.vllm_config.cache_config.block_size = min(
                g.kv_cache_spec.block_size for g in kv_cache_groups
            )

        self.worker.initialize_from_config(kv_cache_configs[0])

        logger.info(
            f"KV cache initialized: num_gpu_blocks={num_blocks}"
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

    async def _join_pipeline(self) -> Optional[int]:
        """Join the pipeline and return the head node's num_gpu_blocks."""
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

            # Extract head's num_gpu_blocks from the response.
            head_num_gpu_blocks = None
            if response.output_data:
                import struct as _struct
                (head_num_gpu_blocks,) = _struct.unpack(
                    "<Q", response.output_data
                )
            return head_num_gpu_blocks
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
