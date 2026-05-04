"""
MoLink Executor for cross-node pipeline parallelism in vLLM v1.

This executor enables distributed pipeline parallelism across multiple physical
nodes using gRPC for communication. It extends the MultiprocExecutor to handle
cross-node tensor transfer and synchronization.
"""

import asyncio
import os
import pickle
import struct
import threading
import traceback
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import cloudpickle
import grpc.aio as aio
import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.executor.multiproc_executor import MultiprocExecutor
from vllm.v1.outputs import ModelRunnerOutput

from molinkv1.service import MolinkService
from molinkv1.utils import (
    extract_ip,
    find_free_port,
    get_grpc_options,
    serialize_metadata,
)
from molinkv1.parallel_state import (
    init_molink_parallel_state,
    destroy_molink_parallel_state,
    is_molink_last_stage,
)
from molinkv1.comm import molink_pb2, molink_pb2_grpc

if TYPE_CHECKING:
    from molinkv1.config import MolinkConfig
    from vllm.v1.core.sched.output import SchedulerOutput

logger = init_logger(__name__)


def _serialize_tensors(tensors_cpu: Dict[str, torch.Tensor]) -> molink_pb2.IntermediateTensors:
    """Serialize a dict of CPU tensors into a protobuf IntermediateTensors message."""
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


class MolinkExecutor(MultiprocExecutor):
    """Executor for cross-node pipeline parallelism using gRPC."""

    supports_pp: bool = True

    @property
    def max_concurrent_batches(self) -> int:
        return 1

    def __init__(self, vllm_config: VllmConfig, monitor_workers: bool = True):
        self.molink_config: "MolinkConfig" = getattr(
            vllm_config, "molink_config", None
        )

        if self.molink_config is None:
            from molinkv1.config import MolinkConfig
            self.molink_config = MolinkConfig(enabled=True)

        # Initialize MoLink parallel state BEFORE parent initialization
        start_layer, end_layer = self.molink_config.get_serving_layers()
        init_molink_parallel_state(
            enabled=True,
            start_layer=start_layer,
            end_layer=end_layer,
        )

        # gRPC server and service
        self.grpc_server: Optional[aio.Server] = None
        self.molink_service: Optional[MolinkService] = None

        # Node information
        self.ip: Optional[str] = None
        self.grpc_port: Optional[int] = None
        self.grpc_address: Optional[str] = None

        # Stub cache for connecting to other nodes
        self._channel_cache: Dict[str, aio.Channel] = {}

        # Event loop for asyncio in separate thread
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_thread: Optional[threading.Thread] = None
        self._shutdown_event = threading.Event()

        # Thread pool for gRPC calls
        self._executor_pool = ThreadPoolExecutor(max_workers=10)

        # Queue to pass scheduler_output from execute_model to sample_tokens.
        self._scheduler_output_queue: deque = deque()

        # Initialize parent executor
        super().__init__(vllm_config, monitor_workers=monitor_workers)

    def _init_executor(self) -> None:
        """Initialize the executor with MoLink support."""
        # First initialize the parent executor (creates local workers)
        super()._init_executor()

        # Initialize MoLink components
        self._init_molink()

    def _is_molink_last_stage(self) -> bool:
        return is_molink_last_stage()

    def _init_molink(self) -> None:
        """Initialize MoLink gRPC server and services."""
        config = self.molink_config

        # Get node IP
        self.ip = extract_ip()

        # Find available gRPC port
        self.grpc_port = find_free_port(
            start_port=config.grpc_port if config.grpc_port > 0 else 50051
        )

        self.grpc_address = f"{self.ip}:{self.grpc_port}"
        logger.info(f"MoLink gRPC server starting at {self.grpc_address}")
        logger.info(
            "DISTRIBUTED SERVICE INFO: If this is the first node of the swarm, "
            f"you can copy the GRPC INFO ({self.grpc_address}) as the initial peer of following nodes"
        )

        # Get layer range
        start_layer, end_layer = config.get_serving_layers()

        # Start event loop in a separate thread
        self._start_event_loop_thread()

        # Schedule gRPC server start in the event loop
        future = asyncio.run_coroutine_threadsafe(
            self._init_grpc_server(start_layer, end_layer), self._event_loop
        )
        future.result(timeout=30)

        # If not head node, join the pipeline
        if not config.is_head_node:
            future = asyncio.run_coroutine_threadsafe(
                self._join_pipeline(), self._event_loop
            )
            try:
                future.result(timeout=30)
            except Exception as e:
                logger.error(f"Failed to join pipeline: {e}")

        logger.info(
            f"MoLink executor initialized. "
            f"Head node: {config.is_head_node}, "
            f"Serving layers: {start_layer}-{end_layer}"
        )

    def _start_event_loop_thread(self) -> None:
        """Start a thread with an event loop for asyncio operations."""
        loop_ready = threading.Event()

        def run_loop():
            self._event_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._event_loop)
            loop_ready.set()
            self._event_loop.run_forever()

        self._loop_thread = threading.Thread(
            target=run_loop, daemon=True, name="MolinkEventLoop"
        )
        self._loop_thread.start()

        # Wait for event loop to be ready
        loop_ready.wait(timeout=10)

    async def _init_grpc_server(self, start_layer: int, end_layer: int) -> None:
        """Initialize and start the gRPC server."""
        config = self.molink_config

        self.grpc_server = aio.server(
            self._executor_pool, options=get_grpc_options(config.max_message_size_mb)
        )

        max_batch_num = 10
        self.molink_service = MolinkService(
            pipeline_size=max_batch_num,
            executor=self,
            head_ip=self.grpc_address,
            start_layer=start_layer,
            end_layer=end_layer,
        )

        molink_pb2_grpc.add_MolinkServiceServicer_to_server(
            self.molink_service, self.grpc_server
        )

        self.grpc_server.add_insecure_port(f"[::]:{self.grpc_port}")

        await self.grpc_server.start()
        logger.info(f"MoLink gRPC server started on port {self.grpc_port}")

    async def _join_pipeline(self) -> None:
        """Join an existing pipeline as a worker node."""
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

    async def _push_intermediate_tensors(
        self,
        tensors: Dict[str, torch.Tensor],
        scheduler_output_bytes: bytes,
        grpc_metadata: Dict[str, Any],
        virtual_engine: int,
        next_server: str,
    ) -> None:
        """Send intermediate tensors to the next pipeline stage via gRPC."""
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

    def _get_stub(self, address: str) -> molink_pb2_grpc.MolinkServiceStub:
        if address not in self._channel_cache:
            channel = aio.insecure_channel(
                address,
                options=get_grpc_options(self.molink_config.max_message_size_mb),
            )
            self._channel_cache[address] = channel
            return molink_pb2_grpc.MolinkServiceStub(channel)
        return molink_pb2_grpc.MolinkServiceStub(self._channel_cache[address])

    async def _push_sampler_output(
        self,
        output_bytes: bytes,
        virtual_engine: int,
        head_server: str,
    ) -> None:
        """Send sampler output to the head node via gRPC."""
        request = molink_pb2.SamplerOutput(
            output_data=output_bytes, virtual_engine=virtual_engine
        )
        stub = self._get_stub(head_server)
        await stub.PushSamplerOutput(request)

    def execute_model(
        self, scheduler_output: "SchedulerOutput", non_block: bool = False
    ) -> ModelRunnerOutput | None | Future[ModelRunnerOutput | None]:
        """Execute the model on local workers.

        For the head node in multi-node mode, we store scheduler_output so
        sample_tokens can send intermediate tensors to the next node.
        """
        if self.molink_config.is_head_node and not self._is_molink_last_stage():
            if scheduler_output.total_num_scheduled_tokens > 0:
                self._scheduler_output_queue.append(scheduler_output)

        return super().execute_model(scheduler_output, non_block)

    def sample_tokens(
        self, grammar_output: Any, non_block: bool = False
    ) -> ModelRunnerOutput | Future[ModelRunnerOutput]:
        """Sample tokens or orchestrate cross-node pipeline."""
        if not (self.molink_config.is_head_node and not self._is_molink_last_stage()):
            return super().sample_tokens(grammar_output, non_block)

        return self._sample_tokens_distributed(non_block)

    def _sample_tokens_distributed(
        self, non_block: bool
    ) -> ModelRunnerOutput | Future[ModelRunnerOutput]:
        """Orchestrate cross-node pipeline and return final ModelRunnerOutput."""

        async def _do_pipeline() -> ModelRunnerOutput:
            scheduler_output = self._scheduler_output_queue.popleft()

            # 1. Get intermediate tensors from local workers.
            results = await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: MultiprocExecutor.collective_rpc(
                    self, "_molink_get_intermediate_tensors"
                ),
            )
            intermediate_tensors = (
                results[0] if isinstance(results, list) else results
            )

            if intermediate_tensors is None:
                logger.error(
                    "[MoLink][PIPELINE] intermediate_tensors is None - "
                    "model runner may not have produced them. "
                    "Falling back to local sample_tokens."
                )
                future = MultiprocExecutor.sample_tokens(
                    self, None, non_block=True
                )
                return await asyncio.get_running_loop().run_in_executor(
                    None, future.result
                )

            # 2. Get pipeline metadata
            grpc_metadata = self.molink_service.topology.get_metadata()
            server_list = grpc_metadata.get("server_list", [])
            virtual_engine = getattr(scheduler_output, "virtual_engine", 0)

            if len(server_list) < 2:
                logger.error(
                    f"[MoLink][PIPELINE] Not enough servers in topology: "
                    f"{server_list}. Falling back to local sample_tokens."
                )
                future = MultiprocExecutor.sample_tokens(
                    self, None, non_block=True
                )
                return await asyncio.get_running_loop().run_in_executor(
                    None, future.result
                )

            # 3. Serialize scheduler_output
            scheduler_output_bytes = cloudpickle.dumps(
                scheduler_output, protocol=pickle.HIGHEST_PROTOCOL
            )

            # 4. Send intermediate tensors to next node (direct gRPC call).
            #    The worker node executes the model directly upon receiving
            #    the data — no separate trigger step is needed.
            next_server = server_list[1]
            await self._push_intermediate_tensors(
                intermediate_tensors.tensors,
                scheduler_output_bytes,
                grpc_metadata,
                virtual_engine,
                next_server,
            )

            # 5. Wait for final result from output_queue
            output_bytes = await self.molink_service.output_queue[
                virtual_engine
            ].get()

            result = cloudpickle.loads(output_bytes)
            return result

        future = asyncio.run_coroutine_threadsafe(
            _do_pipeline(), self._event_loop
        )
        if non_block:
            return future
        else:
            return future.result()

    def shutdown(self) -> None:
        """Shutdown the executor and clean up resources."""
        destroy_molink_parallel_state()

        # Stop gRPC server and close channels in event loop
        if self._event_loop and self._event_loop.is_running():
            future = asyncio.run_coroutine_threadsafe(
                self._async_shutdown(), self._event_loop
            )
            try:
                future.result(timeout=10)
            except Exception as e:
                logger.warning(f"Error during async shutdown: {e}")

        # Stop event loop
        if self._event_loop and self._event_loop.is_running():
            self._event_loop.call_soon_threadsafe(self._event_loop.stop)

        if self._loop_thread and self._loop_thread.is_alive():
            self._loop_thread.join(timeout=5)

        self._executor_pool.shutdown(wait=False)

        super().shutdown()
        logger.info("MoLink executor shutdown complete")

    def get_communication_metrics(self) -> dict:
        return {
            "service_metrics": self.molink_service.get_metrics() if self.molink_service else [],
            "node": self.grpc_address,
            "is_head": self.molink_config.is_head_node,
        }

    def reset_communication_metrics(self):
        if self.molink_service:
            self.molink_service.reset_metrics()

    def _flush_metrics_to_file(self):
        import json
        import tempfile
        data = self.get_communication_metrics()
        path = os.path.join(tempfile.gettempdir(), f"molink_metrics_{self.grpc_port}.json")
        try:
            tmp_path = path + ".tmp"
            with open(tmp_path, "w") as f:
                json.dump(data, f)
            os.replace(tmp_path, path)
        except Exception:
            pass

    async def _async_shutdown(self) -> None:
        if self.grpc_server:
            await self.grpc_server.stop(grace=5)
        for channel in self._channel_cache.values():
            await channel.close()
