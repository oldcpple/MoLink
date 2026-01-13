"""
Async tensor delivery for MoLink cross-node pipeline parallelism.

This module provides async delivery of intermediate tensors and sampler outputs
across nodes using gRPC. The delivery is done in a separate process to overlap
computation with communication.
"""

import asyncio
import io
import multiprocessing as mp
import traceback
from typing import Any, Dict, Optional
import grpc.aio as aio
import torch
from vllm.logger import init_logger
from molinkv1.comm import molink_pb2, molink_pb2_grpc
from molinkv1.utils import get_grpc_options, serialize_metadata

logger = init_logger(__name__)

mp.set_start_method('spawn', force=True)
class TensorDeliveryProcess(mp.Process):
    """Background process for async tensor delivery.

    This process handles serialization and transmission of intermediate
    tensors and sampler outputs to other nodes in the pipeline. Running
    in a separate process allows overlapping of communication with
    computation on the main process.
    """

    def __init__(self, max_message_size_mb: int = 200):
        """Initialize the delivery process.

        Args:
            max_message_size_mb: Maximum gRPC message size in MB.
        """
        super().__init__(daemon=True, name="MolinkTensorDelivery")

        self.max_message_size_mb = max_message_size_mb

        # Queue for pending deliveries
        self.delivery_queue: mp.Queue = mp.Queue(maxsize=200)

        # Shutdown event
        self._shutdown = mp.Event()

    def run(self):
        """Main loop for the delivery process."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # Cache for gRPC channels and stubs
        channel_cache: Dict[str, aio.Channel] = {}
        stub_cache: Dict[str, molink_pb2_grpc.MolinkServiceStub] = {}

        def get_stub(address: str) -> molink_pb2_grpc.MolinkServiceStub:
            if address not in stub_cache:
                channel = aio.insecure_channel(
                    address, options=get_grpc_options(self.max_message_size_mb)
                )
                channel_cache[address] = channel
                stub_cache[address] = molink_pb2_grpc.MolinkServiceStub(channel)
            return stub_cache[address]

        async def deliver_intermediate_tensors(
            intermediate_tensors_cpu: Dict[str, torch.Tensor],
            scheduler_output_bytes: bytes,
            grpc_metadata: Dict[str, Any],
            virtual_engine: int,
            next_server: str,
        ):
            """Deliver intermediate tensors to the next pipeline stage."""
            try:
                # logger.info(
                #     f"[MoLink][VE{virtual_engine}][DELIVERY] Starting deliver_intermediate_tensors to {next_server}"
                # )

                # Serialize tensors
                grpc_tensors = molink_pb2.IntermediateTensors()
                for key, tensor in intermediate_tensors_cpu.items():
                    buffer = io.BytesIO()
                    torch.save(tensor, buffer)
                    tensor_bytes = buffer.getvalue()
                    grpc_tensors.tensors.append(
                        molink_pb2.TensorEntry(key=key, tensor_data=tensor_bytes)
                    )
                    # logger.info(
                    #     f"[MoLink][VE{virtual_engine}][DELIVERY] Serialized tensor '{key}': shape={tensor.shape}, size={len(tensor_bytes)} bytes"
                    # )

                # logger.info(
                #     f"[MoLink][VE{virtual_engine}][DELIVERY] Creating gRPC request..."
                # )
                request = molink_pb2.GrpcRequestData(
                    scheduler_output=scheduler_output_bytes,
                    intermediate_tensors=grpc_tensors,
                    grpc_metadata=serialize_metadata(grpc_metadata),
                    virtual_engine=virtual_engine,
                )

                # logger.info(
                #     f"[MoLink][VE{virtual_engine}][DELIVERY] Sending PushIntermediateTensors to {next_server}..."
                # )
                stub = get_stub(next_server)
                response = await stub.PushIntermediateTensors(request)
                # logger.info(
                #     f"[MoLink][VE{virtual_engine}][DELIVERY] PushIntermediateTensors completed, response: {response.res}"
                # )

            except Exception as e:
                logger.error(
                    f"[MoLink][DELIVERY] Error delivering intermediate tensors: {e}"
                )
                traceback.print_exc()

        async def deliver_sampler_output(
            output_bytes: bytes, virtual_engine: int, head_server: str
        ):
            """Deliver sampler output to the head node."""
            try:
                # logger.info(
                #     f"[MoLink][VE{virtual_engine}][DELIVERY] Starting deliver_sampler_output to {head_server}"
                # )
                # logger.info(
                #     f"[MoLink][VE{virtual_engine}][DELIVERY] Output size: {len(output_bytes)} bytes"
                # )

                request = molink_pb2.SamplerOutput(
                    output_data=output_bytes, virtual_engine=virtual_engine
                )

                # logger.info(
                #     f"[MoLink][VE{virtual_engine}][DELIVERY] Sending PushSamplerOutput to {head_server}..."
                # )
                stub = get_stub(head_server)
                response = await stub.PushSamplerOutput(request)
                # logger.info(
                #     f"[MoLink][VE{virtual_engine}][DELIVERY] PushSamplerOutput completed, response: {response.res}"
                # )

            except Exception as e:
                logger.error(f"[MoLink][DELIVERY] Error delivering sampler output: {e}")
                traceback.print_exc()

        async def consumer_loop():
            """Consumer loop for processing delivery requests."""
            while not self._shutdown.is_set():
                try:
                    # Non-blocking get with timeout
                    item = await loop.run_in_executor(
                        None, lambda: self.delivery_queue.get(timeout=0.1)
                    )
                except Exception:
                    continue

                push_type, data = item

                if push_type == "next":
                    (
                        intermediate_tensors_cpu,
                        scheduler_output_bytes,
                        grpc_metadata,
                        virtual_engine,
                        next_server,
                    ) = data
                    asyncio.create_task(
                        deliver_intermediate_tensors(
                            intermediate_tensors_cpu,
                            scheduler_output_bytes,
                            grpc_metadata,
                            virtual_engine,
                            next_server,
                        )
                    )

                elif push_type == "head":
                    output_bytes, virtual_engine, head_server = data
                    asyncio.create_task(
                        deliver_sampler_output(
                            output_bytes, virtual_engine, head_server
                        )
                    )

        async def main():
            await consumer_loop()
            # Cleanup channels
            for channel in channel_cache.values():
                await channel.close()

        try:
            loop.run_until_complete(main())
        except KeyboardInterrupt:
            pass
        finally:
            loop.close()

    def stop(self):
        """Stop the delivery process."""
        self._shutdown.set()


class TensorDeliveryManager:
    """Manager for async tensor delivery.

    This class provides a high-level interface for delivering tensors
    and outputs to other nodes in the pipeline.
    """

    def __init__(self, max_message_size_mb: int = 200):
        """Initialize the delivery manager.

        Args:
            max_message_size_mb: Maximum gRPC message size in MB.
        """
        self.max_message_size_mb = max_message_size_mb
        self._process: Optional[TensorDeliveryProcess] = None

    def start(self):
        """Start the delivery process."""
        if self._process is None or not self._process.is_alive():
            self._process = TensorDeliveryProcess(self.max_message_size_mb)
            self._process.start()
            logger.info("Tensor delivery process started")

    def stop(self):
        """Stop the delivery process."""
        if self._process is not None:
            self._process.stop()
            self._process.join(timeout=5)
            if self._process.is_alive():
                self._process.terminate()
            self._process = None
            logger.info("Tensor delivery process stopped")

    def deliver_to_next(
        self,
        intermediate_tensors: Dict[str, torch.Tensor],
        scheduler_output_bytes: bytes,
        grpc_metadata: Dict[str, Any],
        virtual_engine: int,
        next_server: str,
    ):
        """Deliver intermediate tensors to the next pipeline stage.

        This method copies tensors to CPU and queues them for async delivery.

        Args:
            intermediate_tensors: Dict of tensor name to GPU tensor.
            scheduler_output_bytes: Serialized scheduler output.
            grpc_metadata: Pipeline metadata.
            virtual_engine: The virtual engine ID.
            next_server: The address of the next server.
        """
        if self._process is None:
            raise RuntimeError("Delivery process not started")

        # Copy tensors to CPU for serialization in delivery process
        tensors_cpu = {k: v.to("cpu") for k, v in intermediate_tensors.items()}

        self._process.delivery_queue.put_nowait(
            (
                "next",
                (
                    tensors_cpu,
                    scheduler_output_bytes,
                    grpc_metadata,
                    virtual_engine,
                    next_server,
                ),
            )
        )

    def deliver_to_head(
        self,
        output_bytes: bytes,
        virtual_engine: int,
        head_server: str
    ):
        """Deliver sampler output to the head node.
        
        Args:
            output_bytes: Serialized ModelRunnerOutput.
            virtual_engine: The virtual engine ID.
            head_server: The address of the head server.
        """
        if self._process is None:
            raise RuntimeError("Delivery process not started")
        
        self._process.delivery_queue.put_nowait((
            'head',
            (output_bytes, virtual_engine, head_server)
        ))
