"""
MoLink gRPC service implementation for cross-node pipeline parallelism.
"""

import asyncio
import io
import traceback
from typing import TYPE_CHECKING, Dict
import torch
from vllm.logger import init_logger
from vllm.sequence import IntermediateTensors
from molinkv1.comm import molink_pb2, molink_pb2_grpc
from molinkv1.utils import PipelineTopology, deserialize_metadata

if TYPE_CHECKING:
    from .executor import MolinkExecutor

logger = init_logger(__name__)


class MolinkService(molink_pb2_grpc.MolinkServiceServicer):
    """gRPC service implementation for MoLink cross-node pipeline parallelism.

    This service handles:
    - Pipeline topology management (joining nodes)
    - Intermediate tensor transfer between pipeline stages
    - Sampler output collection at head node
    - Worker step execution triggers
    """

    def __init__(
        self,
        pipeline_size: int,
        executor: "MolinkExecutor",
        head_ip: str,
        start_layer: int,
        end_layer: int,
    ):
        """Initialize the MoLink service.

        Args:
            pipeline_size: Maximum number of concurrent batches/virtual engines.
            executor: The executor that owns this service.
            head_ip: The IP:port of this node.
            start_layer: First layer this node handles.
            end_layer: Last layer this node handles.
        """
        self.executor = executor
        self.pipeline_size = pipeline_size

        # Queues for inter-stage communication
        # input_queue: receives (scheduler_output, intermediate_tensors, grpc_metadata)
        # output_queue: receives final ModelRunnerOutput
        self.input_queue = [asyncio.Queue() for _ in range(pipeline_size)]
        self.output_queue = [asyncio.Queue() for _ in range(pipeline_size)]

        # Lock for pipeline execution
        self.pp_lock = asyncio.Lock()

        # Pipeline topology
        self.topology = PipelineTopology(head_ip, start_layer, end_layer)

        logger.info(f"MoLink service initialized for node {head_ip}")

    async def JoinPipeline(
        self, request: molink_pb2.NodeInfo, context
    ) -> molink_pb2.GrpcResponseData:
        """Handle a new node joining the pipeline.

        Args:
            request: NodeInfo containing the joining node's information.
            context: gRPC context.

        Returns:
            GrpcResponseData indicating success or failure.
        """
        try:
            node_ip = request.ip
            start_layer = request.start_layer
            end_layer = request.end_layer

            self.topology.add_node(node_ip, start_layer, end_layer)

            logger.info(
                f"Node {node_ip} joined pipeline " f"(layers {start_layer}-{end_layer})"
            )

            return molink_pb2.GrpcResponseData(res=1)

        except Exception as e:
            logger.error(f"Error in JoinPipeline: {e}")
            traceback.print_exc()
            return molink_pb2.GrpcResponseData(res=0, error_message=str(e))

    async def GetTopology(
        self, request: molink_pb2.HealthCheckRequest, context
    ) -> molink_pb2.PipelineTopology:
        """Get the current pipeline topology.

        Args:
            request: Health check request (empty).
            context: gRPC context.

        Returns:
            PipelineTopology containing all nodes in the pipeline.
        """
        nodes = []
        for node in self.topology.node_pool:
            nodes.append(
                molink_pb2.NodeInfo(
                    ip=node["ip"],
                    start_layer=node["start_layer"],
                    end_layer=node["end_layer"],
                )
            )

        return molink_pb2.PipelineTopology(nodes=nodes)

    async def PushIntermediateTensors(
        self, request: molink_pb2.GrpcRequestData, context
    ) -> molink_pb2.GrpcResponseData:
        """Receive intermediate tensors from the previous pipeline stage.

        Args:
            request: GrpcRequestData containing scheduler output and tensors.
            context: gRPC context.

        Returns:
            GrpcResponseData indicating success or failure.
        """
        try:
            virtual_engine = request.virtual_engine
            # logger.info(
            #     f"[MoLink][VE{virtual_engine}][SERVICE] PushIntermediateTensors called"
            # )

            # Store raw bytes for deferred deserialization
            scheduler_output_bytes = request.scheduler_output
            # logger.info(
            #     f"[MoLink][VE{virtual_engine}][SERVICE] Received scheduler output: {len(scheduler_output_bytes)} bytes"
            # )

            # Store tensor bytes (deserialization will be done in worker)
            intermediate_tensors_bytes = {}
            for entry in request.intermediate_tensors.tensors:
                key = entry.key
                byte_data = entry.tensor_data
                intermediate_tensors_bytes[key] = byte_data
            # logger.info(
            #     f"[MoLink][VE{virtual_engine}][SERVICE] Received {len(intermediate_tensors_bytes)} tensors"
            # )

            # Parse grpc metadata
            grpc_metadata = deserialize_metadata(request.grpc_metadata)
            # logger.info(
            #     f"[MoLink][VE{virtual_engine}][SERVICE] Parsed grpc metadata: {list(grpc_metadata.keys())}"
            # )

            # Put into input queue for processing
            await self.input_queue[virtual_engine].put(
                (
                    scheduler_output_bytes,
                    intermediate_tensors_bytes,
                    grpc_metadata,
                )
            )
            # logger.info(
            #     f"[MoLink][VE{virtual_engine}][SERVICE] Data placed in input queue (size: {self.input_queue[virtual_engine].qsize()})"
            # )

            return molink_pb2.GrpcResponseData(res=1)

        except Exception as e:
            logger.error(f"[MoLink][SERVICE] Error in PushIntermediateTensors: {e}")
            traceback.print_exc()
            return molink_pb2.GrpcResponseData(res=0, error_message=str(e))

    async def PushSamplerOutput(
        self, request: molink_pb2.SamplerOutput, context
    ) -> molink_pb2.GrpcResponseData:
        """Receive sampler output from the last pipeline stage.

        This is called on the head node when the last stage completes
        processing and has the final output.

        Args:
            request: SamplerOutput containing the model output.
            context: gRPC context.

        Returns:
            GrpcResponseData indicating success or failure.
        """
        try:
            virtual_engine = request.virtual_engine
            # logger.info(
            #     f"[MoLink][VE{virtual_engine}][SERVICE] PushSamplerOutput called"
            # )

            # Store raw bytes - will be deserialized by the executor
            output_bytes = request.output_data
            # logger.info(
            #     f"[MoLink][VE{virtual_engine}][SERVICE] Received sampler output: {len(output_bytes)} bytes"
            # )

            # Put into output queue for the head node to collect
            await self.output_queue[virtual_engine].put(output_bytes)
            # logger.info(
            #     f"[MoLink][VE{virtual_engine}][SERVICE] Output placed in queue (size: {self.output_queue[virtual_engine].qsize()})"
            # )

            return molink_pb2.GrpcResponseData(res=1)

        except Exception as e:
            logger.error(f"[MoLink][SERVICE] Error in PushSamplerOutput: {e}")
            traceback.print_exc()
            return molink_pb2.GrpcResponseData(res=0, error_message=str(e))

    async def ExecuteWorkerStep(
        self, request: molink_pb2.GrpcTriggerRequest, context
    ) -> molink_pb2.GrpcResponseData:
        """Execute a forward step on this worker node.

        This is called by the head node to trigger execution on worker nodes.
        The worker should already have received intermediate tensors via
        PushIntermediateTensors before this is called.

        Args:
            request: GrpcTriggerRequest containing the virtual engine ID.
            context: gRPC context.

        Returns:
            GrpcResponseData indicating success or failure.
        """
        try:
            virtual_engine = request.virtual_engine
            # logger.info(
            #     f"[MoLink][VE{virtual_engine}][WORKER] ExecuteWorkerStep called"
            # )

            # Get data from input queue (with timeout to detect issues)
            # logger.info(
            #     f"[MoLink][VE{virtual_engine}][WORKER] Waiting for data from input queue..."
            # )
            try:
                scheduler_output_bytes, intermediate_tensors_bytes, grpc_metadata = (
                    await asyncio.wait_for(
                        self.input_queue[virtual_engine].get(), timeout=10.0
                    )
                )
                # logger.info(
                #     f"[MoLink][VE{virtual_engine}][WORKER] Got data from input queue: scheduler={len(scheduler_output_bytes)} bytes, tensors={len(intermediate_tensors_bytes)} items"
                # )
            except asyncio.TimeoutError:
                logger.error(
                    f"[MoLink][VE{virtual_engine}][WORKER] TIMEOUT waiting for input queue! Queue size: {self.input_queue[virtual_engine].qsize()}"
                )
                raise

            # Deserialize tensors in thread pool to not block event loop
            # logger.info(
            #     f"[MoLink][VE{virtual_engine}][WORKER] Deserializing tensors..."
            # )

            def deserialize_tensors(
                tensor_bytes: Dict[str, bytes],
            ) -> IntermediateTensors:
                tensors = {}
                for key, byte_data in tensor_bytes.items():
                    tensor = torch.load(io.BytesIO(byte_data), map_location="cuda")
                    tensors[key] = tensor
                return IntermediateTensors(tensors=tensors)

            intermediate_tensors = await asyncio.to_thread(
                deserialize_tensors, intermediate_tensors_bytes
            )
            # logger.info(
            #     f"[MoLink][VE{virtual_engine}][WORKER] Deserialized {len(intermediate_tensors.tensors)} tensors"
            # )

            # Execute the model step
            # logger.info(
            #     f"[MoLink][VE{virtual_engine}][WORKER] Executing worker step..."
            # )
            async with self.pp_lock:
                result = await self.executor.execute_worker_step(
                    scheduler_output_bytes,
                    intermediate_tensors,
                    grpc_metadata,
                    virtual_engine,
                )
            # logger.info(
            #     f"[MoLink][VE{virtual_engine}][WORKER] Worker step completed, result type: {type(result).__name__}"
            # )

            return molink_pb2.GrpcResponseData(res=1)

        except Exception as e:
            logger.error(f"[MoLink][WORKER] Error in ExecuteWorkerStep: {e}")
            traceback.print_exc()
            return molink_pb2.GrpcResponseData(res=0, error_message=str(e))

    async def HealthCheck(
        self, request: molink_pb2.HealthCheckRequest, context
    ) -> molink_pb2.HealthCheckResponse:
        """Health check endpoint.

        Args:
            request: HealthCheckRequest (empty).
            context: gRPC context.

        Returns:
            HealthCheckResponse indicating the service is healthy.
        """
        return molink_pb2.HealthCheckResponse(status="healthy")
