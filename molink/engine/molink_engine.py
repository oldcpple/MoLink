
from typing import (Any, AsyncGenerator, Callable, Coroutine, Dict, Iterable,
                    List, Mapping, Optional, Set, Tuple, Type, Union, overload)
import asyncio
import copy
import torch
import random
from functools import partial
from weakref import ReferenceType
from vllm.config import VllmConfig
import vllm.envs as envs
from vllm.engine.llm_engine import SchedulerOutputState, SchedulerContext
from vllm.executor.executor_base import ExecutorBase
from vllm.engine.async_llm_engine import AsyncLLMEngine, _AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.metrics_types import StatLoggerBase
from vllm.usage.usage_lib import UsageContext
from vllm.outputs import PoolingRequestOutput, RequestOutput
from vllm.sequence import ExecuteModelRequest
from vllm.logger import init_logger
from vllm.engine.async_timeout import asyncio_timeout
from vllm.utils import (Counter, Device, deprecate_kwargs,
                        resolve_obj_by_qualname, weak_bind)
from vllm.core.scheduler import ScheduledSequenceGroup
from vllm.sampling_params import SamplingParams
from vllm.sequence import SequenceGroupMetadata
from molink.config import MolinkConfig, PipelineConfig
from molink.executor.mp_distributed_executor import MolinkMultiprocessingDistributedExecutor
from .arg_utils import MolinkEngineArgs
import molink.distributed.parallel_state as P
import vllm.distributed.utils as U
import time
from molink.core.scheduler import MolinkScheduler

logger = init_logger(__name__)
ENGINE_ITERATION_TIMEOUT_S = envs.VLLM_ENGINE_ITERATION_TIMEOUT_S


class _MolinkEngine(_AsyncLLMEngine):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_batch_num = 10
        self.scheduler = [
            MolinkScheduler(
                self.scheduler_config, self.cache_config, self.lora_config,
                self.parallel_config.pipeline_parallel_size,
                self.async_callbacks[v_id]
                if self.model_config.use_async_output_proc else None)
            for v_id in range(self.parallel_config.pipeline_parallel_size)
        ]

        self.cached_scheduler_outputs = [
            SchedulerOutputState()
            for _ in range(self.max_batch_num)
        ]

        self.scheduler_contexts = [
            SchedulerContext(multi_step_stream_outputs=self.scheduler_config.
                             multi_step_stream_outputs)
            for _ in range(self.max_batch_num)
        ]

        if self.model_config.use_async_output_proc:
            process_model_outputs = weak_bind(self._process_model_outputs)

            self.async_callbacks = [
                partial(process_model_outputs,
                        ctx=self.scheduler_contexts[v_id])
                for v_id in range(self.max_batch_num)
            ]
        else:
            self.async_callbacks = []

        self.profile_data = {'prefill' : {}, 'decode' : {}}
        self.prerun_profile()

    async def step_async(
        self, virtual_engine: int, ctx_idx: int
    ) -> List[Union[RequestOutput, PoolingRequestOutput]]:
        """Performs one decoding iteration and returns newly generated results.
        The workers are ran asynchronously if possible.

        This function performs one decoding iteration of the engine. It first
        schedules the sequences to be executed in the next iteration and the
        token blocks to be swapped in/out/copy. Then, it executes the model
        and updates the scheduler with the model outputs. Finally, it decodes
        the sequences and returns the newly generated results.
        """
        # these are cached outputs from previous iterations. None if on first
        # iteration
        cached_outputs = self.cached_scheduler_outputs[ctx_idx]

        seq_group_metadata_list = None #cached_outputs.seq_group_metadata_list
        scheduler_outputs = None #cached_outputs.scheduler_outputs
        allow_async_output_proc = None #cached_outputs.allow_async_output_proc

        ctx = self.scheduler_contexts[ctx_idx]

        # Clear outputs for each new scheduler iteration
        ctx.request_outputs.clear()

        # skip the scheduler if there are any remaining steps in the seq groups.
        # This ensures that the scheduler is only called again when the current
        # batch has completed.
        if not self._has_remaining_steps(seq_group_metadata_list):

            (seq_group_metadata_list, scheduler_outputs,
             allow_async_output_proc
             ) = self.scheduler[virtual_engine].schedule()
            
            ctx.seq_group_metadata_list = seq_group_metadata_list
            ctx.scheduler_outputs = scheduler_outputs


            finished_requests_ids = self.scheduler[
                virtual_engine].get_and_reset_finished_requests_ids()

            # Maybe switch from async mode to sync mode
            if not allow_async_output_proc and len(ctx.output_queue) > 0:
                self._process_model_outputs(ctx=ctx)

            if (self.scheduler_config.is_multi_step
                    and scheduler_outputs.num_lookahead_slots > 0):
                # cache the scheduler outputs for the next iteration if we have
                # lookahead slots
                self._cache_scheduler_outputs_for_multi_step(
                    ctx_idx, seq_group_metadata_list, scheduler_outputs,
                    allow_async_output_proc)
        else:
            finished_requests_ids = list()

        if scheduler_outputs.is_empty():
            await asyncio.sleep(0)
            return ctx.request_outputs



        assert seq_group_metadata_list is not None
        assert scheduler_outputs is not None

        if not scheduler_outputs.is_empty():

            # Check if we have a cached last_output from the previous iteration.
            # For supporting PP this is probably the best way to pass the
            # sampled_token_ids, as a separate broadcast over all the PP stages
            # will cause one virtual engine's microbatch to block the pipeline.
            last_sampled_token_ids = \
                self._get_last_sampled_token_ids(ctx_idx)


            execute_model_req = ExecuteModelRequest(
                seq_group_metadata_list=seq_group_metadata_list,
                blocks_to_swap_in=scheduler_outputs.blocks_to_swap_in,
                blocks_to_swap_out=scheduler_outputs.blocks_to_swap_out,
                blocks_to_copy=scheduler_outputs.blocks_to_copy,
                virtual_engine=ctx_idx,
                num_lookahead_slots=scheduler_outputs.num_lookahead_slots,
                running_queue_size=scheduler_outputs.running_queue_size,
                finished_requests_ids=finished_requests_ids,
                # We use ExecuteModelRequest to pass the last sampled_token_ids
                # to each of the non-last PP stages for in-place prepare_input.
                last_sampled_token_ids=last_sampled_token_ids)
            

            record_seq_groups = []
            for sg in scheduler_outputs.scheduled_seq_groups:
                record_seq_groups.append(sg)
                
            # Execute the model.
            outputs = await self.model_executor.execute_model_async(
                execute_model_req)
            
            if len(scheduler_outputs.scheduled_seq_groups) <= 0:
                scheduler_outputs.scheduled_seq_groups.extend(record_seq_groups)
            
            
            
            # we set it to None during execution
            if allow_async_output_proc:
                execute_model_req.async_callback = self.async_callbacks[
                        ctx_idx]
                execute_model_req.async_callback()


            # we need to do this here so that last step's sampled_token_ids can
            # be passed to the next iteration for PP.
            if self.scheduler_config.is_multi_step:
                self._update_cached_scheduler_output(ctx_idx, outputs)
        else:
            if len(ctx.output_queue) > 0:
                self._process_model_outputs(ctx=ctx)
            outputs = []

        # Finish the current step for all the sequence groups.
        if self.scheduler_config.is_multi_step:
            for seq_group in seq_group_metadata_list:
                seq_group.finish_step()


        if not self._has_remaining_steps(seq_group_metadata_list):
            # Clear the cache if we have finished all the steps
            if self.scheduler_config.is_multi_step:
                self.cached_scheduler_outputs[
                    ctx_idx] = SchedulerOutputState()

            # is_first_step_output is True only when the num_steps of all
            # the sequences are 1. When the num_steps > 1,
            # multi_step_model_runner does the first-step output append.
            is_first_step_output: bool = False if not seq_group_metadata_list \
                else seq_group_metadata_list[0].state.num_steps == 1
            
            '''
            if len(seq_group_metadata_list) > len(scheduler_outputs.scheduled_seq_groups):
                seq_group_metadata_list = []
            else:
                scheduler_outputs.scheduled_seq_groups = []
            '''
              
            #scheduler_outputs.scheduled_seq_groups.append(record)

            ctx.append_output(outputs=outputs,
                              seq_group_metadata_list=seq_group_metadata_list,
                              scheduler_outputs=scheduler_outputs,
                              is_async=allow_async_output_proc,
                              is_last_step=True,
                              is_first_step_output=is_first_step_output)
            

            if outputs and allow_async_output_proc:
                assert len(
                    outputs
                ) == 1, "Async postprocessor expects only a single output set"
                self._advance_to_next_step(
                    outputs[0], seq_group_metadata_list,
                    scheduler_outputs.scheduled_seq_groups)

            if not allow_async_output_proc:
                self._process_model_outputs(ctx=ctx)

                # Log stats.
                self.do_log_stats(scheduler_outputs, outputs)

                # Tracing
                self.do_tracing(scheduler_outputs)

            else:
                self._process_model_outputs(ctx=ctx)
            
            self.mark_seq_as_schedule_free(seq_group_metadata_list)


        else:
            # mark seq_group as schedule-free
            # Multi-step case
            return ctx.request_outputs

        if not self.has_unfinished_requests():
            # Drain async postprocessor (if exists)
            if len(ctx.output_queue) > 0:
                self._process_model_outputs(ctx=ctx)
            assert len(ctx.output_queue) == 0

        # mark seq_group as schedule-free


        return ctx.request_outputs
    
    def mark_seq_as_schedule_free(self, seq_group_metadata_list: list):
        for seq_group in seq_group_metadata_list:
            request_id = seq_group.request_id
            self.scheduler[0]._mark_seq_as_schedule_free(request_id)

    def generate_profile_data(self, is_prefill, seq_len, batch_size):
        if is_prefill:
            pass
        else:
            pass
        

    def prerun_profile(self):
        prefill_batched_token_list = [10, 50, 100, 300, 500, 1000, 2000, 3000, 5000]
        decode_batch_size_list = [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

        sampling_params = \
                SamplingParams(top_p=0.99, top_k=self.vocab_size - 1)
        first_layer, last_layer = U.get_pp_indices(1, 1, 1)
        last_layer -= 1
        num_layers = last_layer - first_layer
        
        # profile prefill
        print('MoLink Engine starts to profile prefill latency...')
        for batched_token_num in prefill_batched_token_list:
            seqs: List[SequenceGroupMetadata] = []
            seq_len = batched_token_num
            dummy_data = self.input_registry \
                .dummy_data_for_profiling(self.model_config,
                                            seq_len,
                                            self.mm_registry)
            seq = SequenceGroupMetadata(
                request_id=str(group_id),
                is_prompt=False,
                seq_data={group_id: dummy_data.seq_data},
                sampling_params=sampling_params,
                block_tables=None,
            )
            seqs.append(seq)
            kv_caches = [
                torch.tensor([], dtype=torch.float32, device=self.device)
                for _ in range(num_layers)
            ]
            model_input = self.prepare_model_input(seqs)

            ts = time.time()
            self.execute_model(model_input, kv_caches)
            torch.cuda.synchronize()
            te = time.time()
            # in ms
            profiled_latency = (te - ts) * 1000
            prefill_table = self.profile_data.get('prefill')
            prefill_table.update({batched_token_num : profiled_latency})

        print('Profile of prefill latency finished.')
        print('Prefill latency stats: ')
        for group, latency in self.profile_data['prefill'].items():
            print(group, latency)

        # decode profile
        print('MoLink Engine starts to profile decode latency...')
        for batch_size in decode_batch_size_list:
            seqs: List[SequenceGroupMetadata] = []
            ctn = 0
            for group_id in range(batch_size):
                seq_len = 1
                dummy_data = self.input_registry \
                    .dummy_data_for_profiling(self.model_config,
                                              seq_len,
                                              self.mm_registry)

                seq = SequenceGroupMetadata(
                    request_id=str(ctn),
                    is_prompt=False,
                    seq_data={ctn: dummy_data.seq_data},
                    sampling_params=sampling_params,
                    block_tables=None,
                )
                seqs.append(seq)
            ctn += 1
            kv_caches = [
                torch.tensor([], dtype=torch.float32, device=self.device)
                for _ in range(num_layers)
            ]
            model_input = self.prepare_model_input(seqs)

            ts = time.time()
            self.execute_model(model_input, kv_caches)
            torch.cuda.synchronize()
            te = time.time()
            # in ms
            profiled_latency = (te - ts) * 1000
            prefill_table = self.profile_data.get('decode')
            prefill_table.update({batched_token_num : profiled_latency})

        print('Profile of decode latency finished.')
        print('Decode latency stats: ')
        for group, latency in self.profile_data['prefill'].items():
            print(group, latency)


class MolinkEngine(AsyncLLMEngine):

    _engine_class: Type[_MolinkEngine] = _MolinkEngine

    def __init__(self, *args, **kwargs):

        config = kwargs.get('vllm_config')
        initial_peer = kwargs.get('initial_peer')
        serving_layers = kwargs.get('serving_layers')
        use_dht = kwargs.get('use_dht')
        port = kwargs.get('port')
        in_autodl = kwargs.get('in_autodl')
        autodl_worker_num = kwargs.get('autodl_worker_num')
        P.USE_DHT = use_dht
        P.NODE_PORT = port
        P.IN_AUTODL = in_autodl
        P.AUTODL_WORKER_NUM = autodl_worker_num
        base_port = 38000
        if autodl_worker_num is not None:
            for i in range(autodl_worker_num):
                P.AUTODL_SERVER_IP_MAP.append(f'localhost:{base_port + i}')

        model_config = config.model_config
        num_all_layers = model_config.hf_config.num_hidden_layers
        self.model_hidden_size = model_config.hf_config.hiddensize
        self.model_type_size = 8

        layers_range = [0, num_all_layers - 1]

        if serving_layers is None or serving_layers == '' or len(serving_layers) <= 0:
            serving_layers = [0, num_all_layers - 1]
        else:
            start, end = serving_layers.split(",")
            start = int(start)
            end = int(end)
            serving_layers = [start, end]

        _is_first_rank = serving_layers[0] == layers_range[0]
        _is_last_rank = serving_layers[1] == layers_range[1]

        def get_pp_indices(a, b, c):
            return (serving_layers[0], serving_layers[1] + 1)
        
        U.get_pp_indices = get_pp_indices

        config.__class__ = MolinkConfig
        pipeline_config = PipelineConfig(_is_first_rank, _is_last_rank, initial_peer = initial_peer, serving_layers = serving_layers)
        config._update_attr(pipeline_config)
        kwargs['vllm_config'] = config

        self.initial_peer = initial_peer
        self.serving_layers = serving_layers
        del kwargs['initial_peer']
        del kwargs['serving_layers']
        del kwargs['use_dht']
        del kwargs['port']
        del kwargs['in_autodl']
        del kwargs['autodl_worker_num']

        super().__init__(*args, **kwargs)
    
    @classmethod
    def _get_executor_cls(cls,
                          engine_config: VllmConfig) -> Type[ExecutorBase]:
        return MolinkMultiprocessingDistributedExecutor
    
    @classmethod
    def from_engine_args(
        cls,
        engine_args: MolinkEngineArgs,
        engine_config: Optional[VllmConfig] = None,
        start_engine_loop: bool = True,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: Optional[Dict[str, StatLoggerBase]] = None,
    ) -> "AsyncLLMEngine":
        """Creates an async LLM engine from the engine arguments."""
        # Create the engine configs.
        if engine_config is None:
            engine_config = engine_args.create_engine_config(usage_context)

        executor_class = cls._get_executor_cls(engine_config)

        # Create the async LLM engine.
        engine = cls(
            vllm_config=engine_config,
            executor_class=executor_class,
            log_requests=not engine_args.disable_log_requests,
            log_stats=not engine_args.disable_log_stats,
            start_engine_loop=start_engine_loop,
            usage_context=usage_context,
            stat_loggers=stat_loggers,
            initial_peer = engine_args.initial_peer,
            serving_layers = engine_args.serving_layers,
            use_dht = engine_args.use_dht,
            port = engine_args.port,
            in_autodl = engine_args.in_autodl,
            autodl_worker_num = engine_args.autodl_worker_num,
        )
        return engine
    
    @classmethod
    def from_vllm_config(
        cls,
        vllm_config: VllmConfig,
        start_engine_loop: bool = True,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: Optional[dict[str, StatLoggerBase]] = None,
        disable_log_requests: bool = False,
        disable_log_stats: bool = False,
        engine_args = None,
    ) -> "AsyncLLMEngine":
        """Create an AsyncLLMEngine from the EngineArgs."""

        return cls(
            vllm_config=vllm_config,
            executor_class=cls._get_executor_cls(vllm_config),
            start_engine_loop=start_engine_loop,
            log_requests=not disable_log_requests,
            log_stats=not disable_log_stats,
            usage_context=usage_context,
            stat_loggers=stat_loggers,
            initial_peer = engine_args.initial_peer,
            serving_layers = engine_args.serving_layers,
            use_dht = engine_args.use_dht,
            port = engine_args.port,
            in_autodl = engine_args.in_autodl,
            autodl_worker_num = engine_args.autodl_worker_num,
        )
    
    @staticmethod
    async def run_engine_loop(engine_ref: ReferenceType):
        """We use a weakref to the engine so that the running loop
        doesn't prevent the engine being garbage collected."""
        engine: Optional[AsyncLLMEngine] = engine_ref()
        if not engine:
            return

        pipeline_parallel_size = \
                engine.engine.parallel_config.pipeline_parallel_size
        has_requests_in_progress = [False] * pipeline_parallel_size

        batch_num = 1

        while True:
            if not any(has_requests_in_progress):
                logger.debug("Waiting for new requests...")
                # Stop the execute model loop in parallel workers until there
                # are more requests to process. This avoids waiting
                # indefinitely in torch.distributed ops which may otherwise
                # timeout, and unblocks the RPC thread in the workers so that
                # they can process any other queued control plane messages,
                # such as add/remove lora adapters.
                await engine.engine.stop_remote_worker_execution_loop_async()
                request_tracker = engine._request_tracker
                # Allow engine to be garbage collected while
                # waiting for new requests
                del engine
                await asyncio.sleep(0)
                if engine_ref() is None:
                    return
                await request_tracker.wait_for_new_requests()
                engine = engine_ref()
                if not engine:
                    return
                logger.debug("Got new requests!")

                batch_num = engine.culculate_batch_num()

                requests_in_progress = [
                    asyncio.create_task(engine.engine_step(0, ve))
                    for ve in range(batch_num)
                ]
                has_requests_in_progress = [True] * batch_num

            # Abort if iteration takes too long due to unrecoverable errors
            # (eg. NCCL timeouts).
            try:
                assert len(requests_in_progress) == len(has_requests_in_progress)
                if batch_num > len(requests_in_progress):
                    cur_len = len(requests_in_progress)
                    for i in range(cur_len, batch_num):
                        requests_in_progress.append(asyncio.create_task(engine.engine_step(0, i)))
                        has_requests_in_progress.append(True)

                async with asyncio_timeout(ENGINE_ITERATION_TIMEOUT_S):
                    done, _ = await asyncio.wait(
                        requests_in_progress,
                        return_when=asyncio.FIRST_COMPLETED)
                    for _ in range(pipeline_parallel_size):
                        await asyncio.sleep(0)
                for task in done:
                    result = task.result()
                    virtual_engine = requests_in_progress.index(task)
                    if not has_requests_in_progress[virtual_engine]:
                        continue


                    if virtual_engine >= batch_num:
                        has_requests_in_progress[virtual_engine] = False

                    else:
                        requests_in_progress[virtual_engine] = (
                            asyncio.create_task(
                                engine.engine_step(0, virtual_engine)))
                        has_requests_in_progress[virtual_engine] = True

                    if virtual_engine == 0:
                        batch_num = engine.culculate_batch_num()


            except asyncio.TimeoutError as exc:
                logger.error(
                    "Engine iteration timed out. This should never happen!")
                engine.set_errored(exc)
                raise
            await asyncio.sleep(0)

    async def engine_step(self, virtual_engine: int, ctx_idx: int) -> bool:
        """Kick the engine to process the waiting requests.

        Returns True if there are in-progress requests."""

        new_requests, aborted_requests = (
            self._request_tracker.get_new_and_aborted_requests())

        for new_request in new_requests:
            # Add the request into the vLLM engine's waiting queue.
            try:
                await self.engine.add_request_async(**new_request)
            except ValueError as e:
                # TODO: use a vLLM specific error for failed validation
                self._request_tracker.process_exception(
                    new_request["request_id"],
                    e,
                    verbose=self.log_requests,
                )

        if aborted_requests:
            await self._engine_abort(aborted_requests)

        request_outputs = await self.engine.step_async(virtual_engine, ctx_idx)

        # Put the outputs into the corresponding streams.
        # If used as a callback, then already invoked inside
        # LLMEngine's _process_model_outputs
        if not self.use_process_request_outputs_callback:
            all_finished = self.process_request_outputs(request_outputs)
        else:
            # For callback case, we only need to detect when all
            # requests are finished
            all_finished = all(request_output.finished
                               for request_output in request_outputs)

        return not all_finished
    
    def culculate_compute_latency(self, num_batched_token, batch_size):
        #decode
        left = -1
        right = -1

        keys = sorted(self.engine.profile_data['decode'].keys())

        if batch_size == 1:
            return self.engine.profile_data['decode'].get(1)

        for i in range(0, len(keys) - 1):
            if keys[i] <= batch_size and keys[i + 1] >= batch_size:
                left = keys[i]
                right = keys[i + 1]
                break
        left_latency = self.engine.profile_data['decode'].get(left)
        right_latency = self.engine.profile_data['decode'].get(right)

        return left_latency + (right_latency - left_latency) * ((batch_size - left) / (right - left))
    

    def culculate_transmission_latency(self, num_batched_token, batch_size):
        # Mbps
        bandwidth = 1000
        # ms
        latency = 20
        # bit
        batch_data_size = self.model_type_size * self.model_hidden_size * num_batched_token * batch_size
        # ms
        return (batch_data_size / 1e6) / (bandwidth * 1e6) + latency 
    
    def culculate_batch_num(self): 
        # equal to pipeline size
        base_batch_num = 2
        num_requests = len(self.engine.scheduler.waiting) + len(self.engine.scheduler.running)
        if num_requests == 1:
            return 1
        if num_requests <= base_batch_num:
            return base_batch_num
        for batch_num in range(base_batch_num + 1, self.engine.max_batch_num + 1):
            single_batch_size = num_requests / batch_num
            single_compute_latency = self.culculate_compute_latency(1, single_batch_size)
            single_transmission_latency = self.culculate_transmission_latency(1, single_batch_size)
            bubble = (base_batch_num) * single_transmission_latency
            if (batch_num - base_batch_num) * single_compute_latency >= bubble:
                return batch_num 
        return self.engine.max_batch_num