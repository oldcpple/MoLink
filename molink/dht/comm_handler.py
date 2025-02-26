from concurrent import futures
import grpc
import time
import multiprocessing as mp
import msgspec
import json
import io
import torch
import asyncio
from vllm.sequence import ExecuteModelRequest, IntermediateTensors
from vllm.distributed import get_pp_group
from .utils import decoding_execute_model_req, decoding_sampler_outputs
from .dht import DHTNode
from molink.dht.proto import comm_pb2, comm_pb2_grpc

class CommService(comm_pb2_grpc.CommService):

    def __init__(self, pipeline_size: int, worker):
        self.bind_worker = worker
        self.input_queue = [asyncio.Queue() for _ in range(pipeline_size)]
        self.output_queue = [asyncio.Queue() for _ in range(pipeline_size)]

    async def PushIntermediateTensors(self, request: comm_pb2.GrpcRequestData):
        try:
            #event, request = await self._handler_event_queue.get()
            execute_model_req = request.execute_model_request
            intermediate_tensors = request.intermediate_tensors
            grpc_metadata = request.grpc_metadata
            virtual_engine = request.virtual_engine
    
            execute_model_req = msgspec.json.decode(execute_model_req)
            execute_model_req = decoding_execute_model_req(execute_model_req)
            temp = IntermediateTensors(tensors={})
            temp = {}
            for it in intermediate_tensors.tensors:
                key = it.key
                byte_tensor = it.tensor_data
                temp.update({key:byte_tensor})

            intermediate_tensors = temp
            grpc_metadata = json.loads(grpc_metadata.decode('utf-8'))
            self.input_queue[virtual_engine].put_nowait((execute_model_req, intermediate_tensors, grpc_metadata))
            return comm_pb2.GrpcResponseData(res = 1)

        except Exception as e:
            print(e)

    async def PushSamplerOutput(self, result: comm_pb2.SamplerOutput):
        virtual_engine = result.virtual_engine
        outputs = msgspec.json.decode(result.output_data)
        outputs = [decoding_sampler_outputs(outputs)]
        self.output_queue[virtual_engine].put_nowait(outputs)
        return comm_pb2.GrpcResponseData(res = 1)

    async def ExecutingWorkerStep(self, request: comm_pb2.GrpcTriggerRequest):

        virtual_engine = request.virtual_engine

        execute_model_req, intermediate_tensors, grpc_metadata = await self.input_queue[virtual_engine].get()
        
        if execute_model_req.async_callback is None:
            execute_model_req.async_callback = self.engine.async_callbacks[0]

        temp = IntermediateTensors(tensors={})
        for k, v in intermediate_tensors.items():
            tensors = torch.load(io.BytesIO(v), map_location='cuda')
            temp.tensors.update({k: tensors})

        intermediate_tensors = temp
        
        pipeline_outputs = await self.bind_worker.execute_model(execute_model_req, intermediate_tensors)

        pipeline_outputs = pipeline_outputs[0]
        
        can_push = not get_pp_group().is_last_rank

        if not can_push and get_pp_group().is_first_rank:
            bytes_sampler_outputs = msgspec.json.encode(pipeline_outputs)
            outputs = msgspec.json.decode(bytes_sampler_outputs)
            outputs = [decoding_sampler_outputs(outputs)]
            return outputs

        if can_push:
            server_list = grpc_metadata.get('server_list')
            if len(server_list) <= 0:
                return
            # ip : grpc_port
            next_server = server_list[0]
            # there's no next server
            server_list = server_list[1:]
            grpc_metadata.update({'server_list' : server_list})
            grpc_metadata = json.dumps(grpc_metadata).encode('utf-8')
            execute_model_req.async_callback = None
            bytes_emr = msgspec.json.encode(execute_model_req)

            intermediate_tensors = pipeline_outputs
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
            '''
            if get_pp_group().is_first_rank:
                grpc_result = self.output_queue[virtual_engine].get()
                return grpc_result
            '''
            return []
        
        # the last server in the pipeline
        # push the result to the head server 
        head_server = grpc_metadata.get('head')
        bytes_sampler_outputs = msgspec.json.encode(pipeline_outputs)
        grpc_output_data = comm_pb2.SamplerOutput(output_data=bytes_sampler_outputs)
        with grpc.insecure_channel(head_server) as channel:
            stub = comm_pb2_grpc.CommServiceStub(channel)
            res = await stub.PushSamplerOutput(grpc_output_data)
        return comm_pb2.GrpcResponseData(res = 1)