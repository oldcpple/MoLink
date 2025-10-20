import msgspec
import json
import io
import torch
import asyncio
import traceback
import time
from grpc import aio
from vllm.distributed import get_pp_group
from .utils import decoding_execute_model_req, decoding_sampler_outputs
from vllm.sequence import IntermediateTensors
from molink.comm.proto import comm_pb2, comm_pb2_grpc
import molink.distributed.parallel_state as P

class CommService(comm_pb2_grpc.CommService):

    def __init__(self, pipeline_size: int, executor):
        self.bind_executor = executor
        self.input_queue = [asyncio.Queue() for _ in range(pipeline_size)]
        self.output_queue = [asyncio.Queue() for _ in range(pipeline_size)]
        self.pp_lock = asyncio.Lock()
        self.node_pool = []
        self.node_info_dict = {}

    async def JoinPipeline(self, request: comm_pb2.NodeInfo, context: aio.ServicerContext):
        try:
            node_ip = request.ip
            start_layer = request.start_layer
            end_layer = request.end_layer
            self.node_pool.append({'ip':node_ip, 'start_layer':start_layer, 'end_layer':end_layer})
            self.node_info_dict.update({node_ip : start_layer})
            return comm_pb2.GrpcResponseData(res = 1)
        
        except Exception as e:
            print('Encounter the following exception: {}'.format(e))
            traceback.print_exc()


    async def PushIntermediateTensors(self, request: comm_pb2.GrpcRequestData, context: aio.ServicerContext):
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
            print('Encounter the following exception: {}'.format(e))
            traceback.print_exc()

    async def PushSamplerOutput(self, result: comm_pb2.SamplerOutput, context: aio.ServicerContext):
        try:
            f = open('worker_trace.log', 'a')
            virtual_engine = result.virtual_engine
            print(f'{virtual_engine} trans starts at {time.time()}', file = f)

            outputs = msgspec.json.decode(result.output_data)
            outputs = [decoding_sampler_outputs(outputs)]
            self.output_queue[virtual_engine].put_nowait(outputs)

            print(f'{virtual_engine} trans ends at {time.time()}', file = f)
            f.close()
            return comm_pb2.GrpcResponseData(res = 1)
        
        except Exception as e:
            print('Encounter the following exception: {}'.format(e))
            traceback.print_exc()


    async def ExecutingWorkerStep(self, request: comm_pb2.GrpcTriggerRequest, context: aio.ServicerContext):

        try:
            virtual_engine = request.virtual_engine
            execute_model_req, intermediate_tensors, grpc_metadata = await self.input_queue[virtual_engine].get()

            f = open('worker_trace.log', 'a')
            print(f'{virtual_engine} recv at {time.time()}', file=f)
            batch_size = len(execute_model_req.seq_group_metadata_list)

            def process_tensors(intermediate_tensors):
                temp = IntermediateTensors(tensors={})
                for k, v in intermediate_tensors.items():
                    tensors = torch.load(io.BytesIO(v), map_location='cuda')
                    temp.tensors.update({k: tensors})
                return temp
            
            intermediate_tensors = await asyncio.to_thread(process_tensors, intermediate_tensors)

            if self.bind_executor.parallel_worker_tasks is None:
                # Start model execution loop running in the parallel workers
                self.bind_executor.parallel_worker_tasks = asyncio.create_task(
                    self.bind_executor._start_worker_execution_loop())

            async with self.pp_lock:
                print(f'{virtual_engine} {batch_size} compute starts at {time.time()}', file=f)
                pipeline_outputs = await self.bind_executor.driver_exec_model(execute_model_req, intermediate_tensors)
                torch.cuda.synchronize()
                print(f'{virtual_engine} {batch_size} compute ends at {time.time()}', file = f)

            f.close()

                
            pipeline_outputs = pipeline_outputs[0]
            
            can_push = not get_pp_group().is_last_rank

            if not can_push and get_pp_group().is_first_rank:
                return comm_pb2.GrpcResponseData(res = 0)

            if can_push:
                if not P.IN_AUTODL:
                    server_list = grpc_metadata.get('server_list')

                    idx_self = 0
                    this_ip = f'{self.bind_executor.ip}:{self.bind_executor.grpc_port}'
                    for server in server_list:
                        if server == this_ip:
                            break
                        idx_self += 1

                    # there's no next server
                    if len(server_list) <= idx_self + 1:
                        return
                    # ip : grpc_port
                    next_server = server_list[idx_self + 1]
                
                else:
                    # if we are in autoDL environment, the grpc server address should be 
                    # mapped to localhost:38000, since no direct connection is allowed
                    next_server = 'localhost:38000'

                intermediate_tensors_cpu = {k: v.to('cpu') for k, v in pipeline_outputs.items()}
                self.bind_executor.mp_deliver.process_queue.put_nowait((intermediate_tensors_cpu, execute_model_req, grpc_metadata, \
                                            virtual_engine, next_server, 'next'))
                return comm_pb2.GrpcResponseData(res = 1)
            
            # the last server in the pipeline
            # push the result to the head server 
            # pipeline_outpus should be type of SamplerOutput

            if not P.IN_AUTODL:
                head_server = grpc_metadata.get('head')
            else :
                head_server = 'localhost:38000'
            self.bind_executor.mp_deliver.process_queue.put_nowait((pipeline_outputs, execute_model_req, grpc_metadata, \
                            virtual_engine, head_server, 'head'))
            
            return comm_pb2.GrpcResponseData(res = 1)
        
        except Exception as e:
            print('Encounter the following exception: {}'.format(e))
            traceback.print_exc()
