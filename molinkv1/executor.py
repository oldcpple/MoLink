"""
MoLink Executor for cross-node pipeline parallelism in vLLM v1.

This executor enables distributed pipeline parallelism across multiple physical
nodes using gRPC for communication. It extends the MultiprocExecutor to handle
cross-node tensor transfer and synchronization.

Key design principles:
1. Each physical node runs with local pp_size=1 (no local NCCL PP)
2. Model layers are explicitly distributed via MoLink config
3. Cross-node communication is handled entirely via gRPC
4. The head node orchestrates execution across all nodes
"""

import asyncio
import pickle
import threading
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Dict, List, Optional
import cloudpickle
import grpc.aio as aio
import msgspec
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.sequence import IntermediateTensors
from vllm.v1.executor.multiproc_executor import MultiprocExecutor
from vllm.v1.outputs import ModelRunnerOutput
from molinkv1.delivery import TensorDeliveryManager
from molinkv1.service import MolinkService
from molinkv1.utils import (
    extract_ip,
    find_free_port,
    get_grpc_options,
)
from .parallel_state import (
    init_molink_parallel_state,
    destroy_molink_parallel_state,
)
from molinkv1.comm import molink_pb2, molink_pb2_grpc

if TYPE_CHECKING:
    from molinkv1.config import MolinkConfig
    from vllm.v1.core.sched.output import SchedulerOutput

logger = init_logger(__name__)


class MolinkExecutor(MultiprocExecutor):
    """Executor for cross-node pipeline parallelism using gRPC.

    This executor extends MultiprocExecutor to support distributed pipeline
    parallelism across multiple physical nodes. Each node runs a subset of
    the model layers, and intermediate tensors are transferred between nodes
    using gRPC.

    The head node orchestrates the execution:
    1. Receives requests and schedules batches
    2. Executes its layers and sends intermediate tensors to the next node
    3. Triggers worker nodes to execute their layers
    4. Receives final output from the last node

    Worker nodes:
    1. Wait for trigger from head node
    2. Receive intermediate tensors from previous node
    3. Execute their layers
    4. Send results to next node or back to head
    """

    supports_pp: bool = True

    def __init__(self, vllm_config: VllmConfig, monitor_workers: bool = True):
        """Initialize the MoLink executor.

        Args:
            vllm_config: The vLLM configuration.
            monitor_workers: Whether to monitor worker processes.
        """
        self.molink_config: "MolinkConfig" = getattr(vllm_config, "molink_config", None)

        if self.molink_config is None:
            # Create a default config if not provided
            from molinkv1.config import MolinkConfig

            self.molink_config = MolinkConfig(enabled=True)

        # Initialize MoLink parallel state BEFORE parent initialization
        # This ensures model layers are correctly distributed
        start_layer, end_layer = self.molink_config.get_serving_layers()
        init_molink_parallel_state(
            enabled=True,
            start_layer=start_layer,
            end_layer=end_layer,
        )

        # gRPC server and service
        self.grpc_server: Optional[aio.Server] = None
        self.molink_service: Optional[MolinkService] = None

        # Tensor delivery manager
        self.delivery_manager: Optional[TensorDeliveryManager] = None

        # Node information
        self.ip: Optional[str] = None
        self.grpc_port: Optional[int] = None
        self.grpc_address: Optional[str] = None

        # Stub cache for connecting to other nodes
        self._stub_cache: Dict[str, molink_pb2_grpc.MolinkServiceStub] = {}
        self._channel_cache: Dict[str, aio.Channel] = {}

        # Event loop for asyncio in separate thread
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_thread: Optional[threading.Thread] = None
        self._shutdown_event = threading.Event()

        # Thread pool for gRPC calls
        self._executor_pool = ThreadPoolExecutor(max_workers=10)

        # Pipeline parallel lock (same as v0 implementation)
        self.pp_lock: Optional[asyncio.Lock] = None

        # Worker tasks
        self.parallel_worker_tasks: Optional[asyncio.Task] = None

        # Cached stubs for pipeline servers
        self._preset_server_list: List[str] = []
        self._stub_list: List[molink_pb2_grpc.MolinkServiceStub] = []
        # Initialize parent executor
        super().__init__(vllm_config, monitor_workers=monitor_workers)

    def _init_executor(self) -> None:
        """Initialize the executor with MoLink support."""
        # First initialize the parent executor (creates local workers)
        super()._init_executor()

        # Initialize MoLink components
        self._init_molink()

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
        future.result(timeout=30)  # Wait for initialization

        # Initialize tensor delivery manager
        self.delivery_manager = TensorDeliveryManager(
            max_message_size_mb=config.max_message_size_mb
        )
        self.delivery_manager.start()

        # If not head node, join the pipeline
        if not config.is_head_node:
            future = asyncio.run_coroutine_threadsafe(
                self._join_pipeline(), self._event_loop
            )
            try:
                future.result(timeout=30)  # Wait for join
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
            self._event_loop.run_until_complete(self._run_until_shutdown())

        self._loop_thread = threading.Thread(
            target=run_loop, daemon=True, name="MolinkEventLoop"
        )
        self._loop_thread.start()

        # Wait for event loop to be ready
        loop_ready.wait(timeout=10)

    async def _run_until_shutdown(self) -> None:
        """Run the event loop until shutdown is signaled."""
        while not self._shutdown_event.is_set():
            await asyncio.sleep(0.1)

    async def _init_grpc_server(self, start_layer: int, end_layer: int) -> None:
        """Initialize and start the gRPC server."""
        config = self.molink_config

        # Create gRPC server
        self.grpc_server = aio.server(
            self._executor_pool, options=get_grpc_options(config.max_message_size_mb)
        )

        # Create MoLink service
        max_batch_num = 10  # Maximum concurrent batches
        self.molink_service = MolinkService(
            pipeline_size=max_batch_num,
            executor=self,
            head_ip=self.grpc_address,
            start_layer=start_layer,
            end_layer=end_layer,
        )

        # Register service
        molink_pb2_grpc.add_MolinkServiceServicer_to_server(
            self.molink_service, self.grpc_server
        )

        # Add port
        self.grpc_server.add_insecure_port(f"[::]:{self.grpc_port}")

        # Start server
        await self.grpc_server.start()
        logger.info(f"MoLink gRPC server started on port {self.grpc_port}")

    async def _join_pipeline(self) -> None:
        """Join an existing pipeline as a worker node."""
        config = self.molink_config
        channel = None

        try:
            # Create channel to initial peer
            channel = aio.insecure_channel(
                config.initial_peer,
                options=get_grpc_options(config.max_message_size_mb),
            )
            stub = molink_pb2_grpc.MolinkServiceStub(channel)

            # Register with the head node
            start_layer, end_layer = config.get_serving_layers()
            node_info = molink_pb2.NodeInfo(
                ip=self.grpc_address, start_layer=start_layer, end_layer=end_layer
            )

            # Await the gRPC call
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

    def _get_stub(self, address: str) -> molink_pb2_grpc.MolinkServiceStub:
        """Get or create a gRPC stub for the given address.

        Args:
            address: The gRPC address (host:port).

        Returns:
            A MolinkServiceStub for the address.
        """
        if address not in self._stub_cache:
            channel = aio.insecure_channel(
                address,
                options=get_grpc_options(self.molink_config.max_message_size_mb),
            )
            self._channel_cache[address] = channel
            self._stub_cache[address] = molink_pb2_grpc.MolinkServiceStub(channel)

        return self._stub_cache[address]

    def _update_stub_list(self, server_list: List[str]) -> None:
        """Update cached stub list if server list changed.

        Args:
            server_list: List of server addresses (excluding head).
        """
        if server_list == self._preset_server_list:
            return

        self._preset_server_list = server_list
        self._stub_list = [self._get_stub(server) for server in server_list]

    def execute_model(
        self, scheduler_output: "SchedulerOutput", non_block: bool = False
    ) -> ModelRunnerOutput | None | Future[ModelRunnerOutput | None]:
        """Execute the model (main entry point).

        For head node: orchestrates the distributed pipeline
        For worker node: should not be called directly (uses service)

        Args:
            scheduler_output: The scheduler output.
            non_block: Whether to return immediately with a Future.

        Returns:
            ModelRunnerOutput or Future[ModelRunnerOutput].
        """
        if not self.molink_config.is_head_node:
            # Worker nodes shouldn't call this directly
            # They receive triggers via gRPC service
            return super().execute_model(scheduler_output, non_block)

        # Head node: orchestrate the pipeline
        if non_block:
            future = asyncio.run_coroutine_threadsafe(
                self._execute_model_async(scheduler_output), self._event_loop
            )
            return future
        else:
            future = asyncio.run_coroutine_threadsafe(
                self._execute_model_async(scheduler_output), self._event_loop
            )
            return future.result()

    async def _execute_model_async(
        self, scheduler_output: "SchedulerOutput"
    ) -> ModelRunnerOutput:
        """Execute the model across the distributed pipeline.

        This method orchestrates execution across all nodes in the pipeline.

        Args:
            scheduler_output: The scheduler output containing batch information.

        Returns:
            ModelRunnerOutput from the final pipeline stage.
        """
        try:
            if self.pp_lock is None:
                self.pp_lock = asyncio.Lock()

            # Get pipeline metadata
            grpc_metadata = self.molink_service.topology.get_metadata()
            server_list = grpc_metadata.get("server_list", [])

            virtual_engine = getattr(scheduler_output, "virtual_engine", 0)

            # Serialize scheduler output using cloudpickle
            scheduler_output_bytes = cloudpickle.dumps(
                scheduler_output, protocol=pickle.HIGHEST_PROTOCOL
            )
            logger.info(
                f"[MoLink][VE{virtual_engine}] Serialized scheduler output: {len(scheduler_output_bytes)} bytes"
            )
            logger.info(
                f"[MoLink][VE{virtual_engine}] Scheduler output: new_reqs={len(scheduler_output.scheduled_new_reqs)}, "
                f"cached_reqs={len(scheduler_output.scheduled_cached_reqs.req_ids)}, "
                f"total_tokens={scheduler_output.total_num_scheduled_tokens}"
            )

            # Skip execution if no tokens to schedule
            if scheduler_output.total_num_scheduled_tokens == 0:
                logger.info(
                    f"[MoLink][VE{virtual_engine}] No tokens to schedule, skipping execution"
                )
                # When total_tokens=0, it means:
                # 1. Request just finished: finished_req_ids is non-empty, the final output
                #    was already returned in the previous step
                # 2. No active requests: both finished_req_ids and running requests are empty
                # In both cases, we need to return an empty ModelRunnerOutput
                if scheduler_output.finished_req_ids:
                    logger.info(
                        f"[MoLink][VE{virtual_engine}] Request(s) finished: {scheduler_output.finished_req_ids}"
                    )
                logger.info(
                    f"[MoLink][VE{virtual_engine}] Returning empty ModelRunnerOutput"
                )

                # Import ModelRunnerOutput here to avoid circular imports
                from vllm.v1.outputs import ModelRunnerOutput

                empty_output = ModelRunnerOutput(
                    req_ids=[],
                    req_id_to_index={},
                    sampled_token_ids=[],
                    logprobs=None,
                    prompt_logprobs_dict={},
                    pooler_output=[],
                    kv_connector_output=None,
                    ec_connector_output=None,
                    num_nans_in_logits=None,
                )
                return empty_output

            # Execute on head node
            head_task = asyncio.create_task(
                self._executing_head_server(
                    scheduler_output, scheduler_output_bytes, grpc_metadata
                )
            )

            # Trigger execution on worker nodes (skip head node at index 0)
            worker_servers = server_list[1:]
            self._update_stub_list(worker_servers)

            trigger_request = molink_pb2.GrpcTriggerRequest(
                virtual_engine=virtual_engine
            )

            # Trigger all worker nodes asynchronously
            # Wrap gRPC calls in async functions for create_task
            async def trigger_worker(stub, request):
                return await stub.ExecuteWorkerStep(request)

            worker_tasks = [
                asyncio.create_task(trigger_worker(stub, trigger_request))
                for stub in self._stub_list
            ]

            # Wait for result from output queue
            output_bytes = await self.molink_service.output_queue[virtual_engine].get()

            # Deserialize result
            if isinstance(output_bytes, bytes):
                result = cloudpickle.loads(output_bytes)
                logger.info(
                    f"[MoLink][VE{virtual_engine}] Deserialized result: {type(result).__name__}"
                )
            else:
                result = output_bytes

            return result

        except Exception as e:
            logger.error(f"Error in execute_model_async: {e}")
            traceback.print_exc()
            raise

    async def _executing_head_server(
        self,
        scheduler_output: "SchedulerOutput",
        scheduler_output_bytes: bytes,
        grpc_metadata: Dict[str, Any],
    ) -> None:
        """Execute model on the head node and handle results.

        Args:
            scheduler_output: The scheduler output.
            scheduler_output_bytes: Serialized scheduler output (pickle format).
            grpc_metadata: Pipeline metadata.
        """
        try:
            virtual_engine = getattr(scheduler_output, "virtual_engine", 0)

            logger.info(f"[MoLink][VE{virtual_engine}][HEAD] Starting head execution")

            # Execute on local workers
            async with self.pp_lock:
                logger.info(
                    f"[MoLink][VE{virtual_engine}][HEAD] Calling _driver_exec_model"
                )
                output = await self._driver_exec_model(scheduler_output)
                logger.info(
                    f"[MoLink][VE{virtual_engine}][HEAD] _driver_exec_model completed, output type: {type(output).__name__}"
                )

            server_list = grpc_metadata.get("server_list", [])

            # If this is the only node, put result directly in output queue
            if len(server_list) <= 1:
                logger.info(
                    f"[MoLink][VE{virtual_engine}][HEAD] Single node, putting result in output queue"
                )
                # Serialize output for consistency
                output_bytes = msgspec.json.encode(output)
                await self.molink_service.output_queue[virtual_engine].put(output_bytes)
                logger.info(
                    f"[MoLink][VE{virtual_engine}][HEAD] Result placed in output queue"
                )
                return

            # Multi-node case: extract and send intermediate tensors
            next_server = server_list[1]
            logger.info(
                f"[MoLink][VE{virtual_engine}][HEAD] Multi-node, next server: {next_server}"
            )

            # Get intermediate tensors
            if isinstance(output, IntermediateTensors):
                intermediate_tensors = output.tensors
                logger.info(
                    f"[MoLink][VE{virtual_engine}][HEAD] Got IntermediateTensors with keys: {list(intermediate_tensors.keys())}"
                )
            elif hasattr(output, "tensors"):
                intermediate_tensors = output.tensors
                logger.info(
                    f"[MoLink][VE{virtual_engine}][HEAD] Got tensors from output.tensors with keys: {list(intermediate_tensors.keys())}"
                )
            else:
                # Head node got ModelRunnerOutput - this means the request finished
                # This can happen when there's a final empty decode step after request completion
                logger.info(
                    f"[MoLink][VE{virtual_engine}][HEAD] Got final output (non-intermediate), type: {type(output)}"
                )
                # Serialize using cloudpickle for consistency
                output_bytes = cloudpickle.dumps(output)
                await self.molink_service.output_queue[virtual_engine].put(output_bytes)
                logger.info(
                    f"[MoLink][VE{virtual_engine}][HEAD] Final output placed in queue"
                )
                return

            # Deliver to next node (async in background process)
            logger.info(
                f"[MoLink][VE{virtual_engine}][HEAD] Delivering {len(intermediate_tensors)} tensors to {next_server}"
            )
            self.delivery_manager.deliver_to_next(
                intermediate_tensors,
                scheduler_output_bytes,
                grpc_metadata,
                virtual_engine,
                next_server,
            )
            logger.info(
                f"[MoLink][VE{virtual_engine}][HEAD] Delivery initiated (async)"
            )

        except Exception as e:
            logger.error(f"[MoLink][HEAD] Error in _executing_head_server: {e}")
            traceback.print_exc()

    async def _driver_exec_model(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> Any:
        """Execute model on local workers.

        Args:
            scheduler_output: The scheduler output.
            intermediate_tensors: Optional intermediate tensors from previous stage.

        Returns:
            Model output or intermediate tensors.
        """
        virtual_engine = getattr(scheduler_output, "virtual_engine", 0)

        logger.info(
            f"[MoLink][VE{virtual_engine}][DRIVER] Calling parent execute_model"
        )

        # For MoLink: if we have intermediate_tensors, we need to pass them to workers
        # We'll use a custom approach: store them in model_runner before execution
        if intermediate_tensors is not None:
            logger.info(
                f"[MoLink][VE{virtual_engine}][DRIVER] Have intermediate tensors with {len(intermediate_tensors.tensors)} items"
            )
            # Store intermediate tensors in a way workers can access
            # Use collective_rpc with custom method to set intermediate tensors first
            logger.info(
                f"[MoLink][VE{virtual_engine}][DRIVER] Setting intermediate tensors on workers"
            )

            # First, send intermediate tensors to all workers
            self.collective_rpc(
                "_molink_set_intermediate_tensors",
                args=(intermediate_tensors,),
            )

        # Now execute the model
        logger.info(f"[MoLink][VE{virtual_engine}][DRIVER] Executing model on workers")
        future = super().execute_model(scheduler_output, non_block=True)

        # Wait for the Future to complete
        logger.info(
            f"[MoLink][VE{virtual_engine}][DRIVER] Waiting for execution to complete"
        )
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, future.result)

        logger.info(
            f"[MoLink][VE{virtual_engine}][DRIVER] Execution completed, result type: {type(result).__name__}"
        )

        return result

    async def execute_worker_step(
        self,
        scheduler_output_bytes: bytes,
        intermediate_tensors: IntermediateTensors,
        grpc_metadata: Dict[str, Any],
        virtual_engine: int,
    ) -> Any:
        """Execute a forward step on a worker node.

        Called by the MoLink service when this node receives a trigger
        to execute its layers.

        Args:
            scheduler_output_bytes: Serialized scheduler output (pickle format).
            intermediate_tensors: Intermediate tensors from previous stage.
            grpc_metadata: Pipeline metadata.
            virtual_engine: The virtual engine ID.

        Returns:
            The output (intermediate tensors or final result).
        """
        try:
            logger.info(
                f"[MoLink][VE{virtual_engine}][WORKER_STEP] Starting execute_worker_step"
            )

            # Deserialize scheduler output using cloudpickle
            scheduler_output = cloudpickle.loads(scheduler_output_bytes)
            logger.info(
                f"[MoLink][VE{virtual_engine}][WORKER_STEP] Deserialized scheduler output"
            )
            logger.info(
                f"[MoLink][VE{virtual_engine}][WORKER_STEP] Scheduler output: new_reqs={len(scheduler_output.scheduled_new_reqs)}, "
                f"cached_reqs={len(scheduler_output.scheduled_cached_reqs.req_ids)}, "
                f"total_tokens={scheduler_output.total_num_scheduled_tokens}"
            )

            # TODO: Set intermediate tensors on workers
            # This requires a mechanism to pass intermediate tensors to the model runner
            # For now, we'll rely on the model receiving them via a different path
            logger.info(
                f"[MoLink][VE{virtual_engine}][WORKER_STEP] Received {len(intermediate_tensors.tensors)} intermediate tensors"
            )

            # Execute on local workers
            logger.info(
                f"[MoLink][VE{virtual_engine}][WORKER_STEP] Calling _driver_exec_model"
            )
            output = await self._driver_exec_model(
                scheduler_output, intermediate_tensors
            )
            logger.info(
                f"[MoLink][VE{virtual_engine}][WORKER_STEP] _driver_exec_model completed, output type: {type(output).__name__}"
            )

            # If output is None, we need to call sample_tokens (for last stage)
            if output is None:
                logger.info(
                    f"[MoLink][VE{virtual_engine}][WORKER_STEP] Output is None, calling sample_tokens"
                )
                # Call sample_tokens via collective_rpc
                future = self.sample_tokens(None, non_block=True)
                loop = asyncio.get_running_loop()
                output = await loop.run_in_executor(None, future.result)
                logger.info(
                    f"[MoLink][VE{virtual_engine}][WORKER_STEP] sample_tokens completed, output type: {type(output).__name__}"
                )

            server_list = grpc_metadata.get("server_list", [])
            logger.info(
                f"[MoLink][VE{virtual_engine}][WORKER_STEP] Pipeline has {len(server_list)} nodes"
            )

            # Find my position in pipeline
            try:
                my_idx = server_list.index(self.grpc_address)
                logger.info(
                    f"[MoLink][VE{virtual_engine}][WORKER_STEP] My position in pipeline: {my_idx}"
                )
            except ValueError:
                logger.error(
                    f"[MoLink][VE{virtual_engine}][WORKER_STEP] Node {self.grpc_address} not found in server list: {server_list}"
                )
                return output

            is_last_stage = my_idx == len(server_list) - 1
            logger.info(
                f"[MoLink][VE{virtual_engine}][WORKER_STEP] Is last stage: {is_last_stage}"
            )

            if is_last_stage:
                # Send final output back to head node
                head_server = grpc_metadata.get("head")
                logger.info(
                    f"[MoLink][VE{virtual_engine}][WORKER_STEP] Last stage, sending final output to head: {head_server}"
                )

                # Check if output is ModelRunnerOutput
                if isinstance(output, ModelRunnerOutput):
                    output_bytes = cloudpickle.dumps(
                        output, protocol=pickle.HIGHEST_PROTOCOL
                    )
                    logger.info(
                        f"[MoLink][VE{virtual_engine}][WORKER_STEP] Serialized ModelRunnerOutput: {len(output_bytes)} bytes"
                    )
                else:
                    logger.error(
                        f"[MoLink][VE{virtual_engine}][WORKER_STEP] ERROR: Last stage output is not ModelRunnerOutput, type: {type(output)}"
                    )
                    # Try to serialize anyway
                    output_bytes = cloudpickle.dumps(
                        output, protocol=pickle.HIGHEST_PROTOCOL
                    )

                self.delivery_manager.deliver_to_head(
                    output_bytes, virtual_engine, head_server
                )
                logger.info(
                    f"[MoLink][VE{virtual_engine}][WORKER_STEP] Final output delivery initiated"
                )
            else:
                # Send intermediate tensors to next node
                next_server = server_list[my_idx + 1]
                logger.info(
                    f"[MoLink][VE{virtual_engine}][WORKER_STEP] Intermediate stage, next server: {next_server}"
                )

                if isinstance(output, IntermediateTensors):
                    intermediate = output.tensors
                    logger.info(
                        f"[MoLink][VE{virtual_engine}][WORKER_STEP] Got IntermediateTensors with {len(intermediate)} items"
                    )
                elif hasattr(output, "tensors"):
                    intermediate = output.tensors
                    logger.info(
                        f"[MoLink][VE{virtual_engine}][WORKER_STEP] Got tensors from output with {len(intermediate)} items"
                    )
                else:
                    intermediate = {"hidden_states": output}
                    logger.info(
                        f"[MoLink][VE{virtual_engine}][WORKER_STEP] Wrapping output in hidden_states"
                    )

                # Serialize scheduler output for forwarding
                logger.info(
                    f"[MoLink][VE{virtual_engine}][WORKER_STEP] Delivering {len(intermediate)} tensors to {next_server}"
                )
                self.delivery_manager.deliver_to_next(
                    intermediate,
                    scheduler_output_bytes,
                    grpc_metadata,
                    virtual_engine,
                    next_server,
                )
                logger.info(
                    f"[MoLink][VE{virtual_engine}][WORKER_STEP] Intermediate tensor delivery initiated"
                )

            logger.info(
                f"[MoLink][VE{virtual_engine}][WORKER_STEP] execute_worker_step completed"
            )
            return output

        except Exception as e:
            logger.error(f"Error in execute_worker_step: {e}")
            traceback.print_exc()
            raise

    def shutdown(self) -> None:
        """Shutdown the executor and clean up resources."""
        # Destroy MoLink parallel state
        destroy_molink_parallel_state()

        # Stop delivery manager
        if self.delivery_manager:
            self.delivery_manager.stop()

        # Stop gRPC server and close channels in event loop
        if self._event_loop and self._event_loop.is_running():
            future = asyncio.run_coroutine_threadsafe(
                self._async_shutdown(), self._event_loop
            )
            try:
                future.result(timeout=10)
            except Exception as e:
                logger.warning(f"Error during async shutdown: {e}")

        # Signal event loop to stop
        self._shutdown_event.set()

        # Wait for event loop thread
        if self._loop_thread and self._loop_thread.is_alive():
            self._loop_thread.join(timeout=5)

        # Shutdown thread pool
        self._executor_pool.shutdown(wait=False)

        # Call parent shutdown
        super().shutdown()

        logger.info("MoLink executor shutdown complete")

    async def _async_shutdown(self) -> None:
        """Async cleanup of gRPC resources."""
        # Stop gRPC server
        if self.grpc_server:
            await self.grpc_server.stop(grace=5)

        # Close gRPC channels
        for channel in self._channel_cache.values():
            await channel.close()
