
from typing import (Dict, List, Optional, Type, Union)
import asyncio
import torch
from functools import partial
from weakref import ReferenceType
from vllm.config import VllmConfig
import vllm.envs as envs
from vllm.engine.llm_engine import SchedulerOutputState, SchedulerContext
from vllm.executor.executor_base import ExecutorBase
from vllm.engine.async_llm_engine import AsyncLLMEngine, _AsyncLLMEngine
from vllm.engine.metrics_types import StatLoggerBase
from vllm.usage.usage_lib import UsageContext
from vllm.outputs import PoolingRequestOutput, RequestOutput
from vllm.sequence import ExecuteModelRequest, SequenceStage
from vllm.logger import init_logger
from vllm.utils import weak_bind
from vllm.sampling_params import SamplingParams
from vllm.sequence import SequenceGroupMetadata
from molink.config import MolinkConfig, PipelineConfig
from molink.executor.mp_distributed_executor import MolinkMultiprocessingDistributedExecutor
from .arg_utils import MolinkEngineArgs
import molink.distributed.parallel_state as P
import vllm.distributed.utils as U
import time
import os
from molink.core.scheduler import MolinkScheduler
from vllm.sequence import (ExecuteModelRequest, ParallelSampleSequenceGroup,
                           PoolingSequenceGroupOutput, Sequence, SequenceGroup,
                           SequenceGroupBase, SequenceGroupMetadata,
                           SequenceGroupOutput, SequenceStatus)
from vllm.engine.output_processor.util import create_output_by_sequence_group
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.outputs import (PoolingRequestOutput, RequestOutput,
                          RequestOutputFactory)
from vllm.sampling_params import RequestOutputKind, SamplingParams

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
        self.profile_data['prefill'] = {10: 20.84064483642578, 50: 23.522377014160156, 100: 28.12337875366211, 
                                        300: 66.10918045043945, 500: 91.38917922973633, 1000: 136.0461711883545, 
                                        2000: 250.7038116455078, 3000: 384.86647605895996, 5000: 670.1204776763916}
        self.profile_data['decode'] = {1: 12.014627456665039, 2: 12.270689010620117, 3: 12.321710586547852, 4: 12.281417846679688, 5: 12.432098388671875, 6: 12.410879135131836, 7: 12.396812438964844, 8: 12.4053955078125, 9: 12.728452682495117, 10: 12.69388198852539, 11: 12.683868408203125, 12: 12.680530548095703, 13: 12.677431106567383, 14: 12.681722640991211, 15: 12.677431106567383, 16: 12.681961059570312, 17: 15.767335891723633, 18: 15.749216079711914, 19: 15.729188919067383, 20: 15.72728157043457, 21: 15.731573104858398, 22: 15.725374221801758, 23: 15.712738037109375, 24: 15.728950500488281, 25: 15.913248062133789, 26: 15.891790390014648, 27: 15.89202880859375, 28: 15.870332717895508, 29: 15.887260437011719, 30: 15.880823135375977, 31: 15.899181365966797, 32: 15.871047973632812, 33: 16.055583953857422, 34: 16.02458953857422, 35: 16.05057716369629, 36: 16.026735305786133, 37: 16.04604721069336, 38: 16.104698181152344, 39: 16.028642654418945, 40: 16.21723175048828, 41: 16.186952590942383, 42: 16.131877899169922, 43: 16.14975929260254, 44: 16.145944595336914, 45: 16.142606735229492, 46: 16.14546775817871, 47: 16.133546829223633, 48: 16.14212989807129, 49: 16.31021499633789, 50: 16.268491744995117, 51: 16.27063751220703, 52: 16.284465789794922, 53: 16.271352767944336, 54: 16.260862350463867, 55: 16.25990867614746, 56: 16.25657081604004, 57: 16.432762145996094, 58: 16.434669494628906, 59: 16.423940658569336, 60: 16.422748565673828, 61: 16.405344009399414, 62: 16.405582427978516, 63: 16.422510147094727, 64: 16.454696655273438, 65: 18.418550491333008, 66: 18.396615982055664, 67: 18.395662307739258, 68: 18.388748168945312, 69: 18.41282844543457, 70: 18.399477005004883, 71: 18.401384353637695, 72: 18.388748168945312, 73: 18.56827735900879, 74: 18.56207847595215, 75: 18.591880798339844, 76: 18.589258193969727, 77: 18.596887588500977, 78: 18.579483032226562, 79: 18.58353614807129, 80: 18.574953079223633, 81: 18.725872039794922, 82: 18.733501434326172, 83: 18.738508224487305, 84: 18.769025802612305, 85: 18.807649612426758, 86: 18.7225341796875, 87: 18.755197525024414, 88: 18.743276596069336, 89: 18.870115280151367, 90: 18.879413604736328, 91: 19.000530242919922, 92: 19.000768661499023, 93: 19.016742706298828, 94: 19.0126895904541, 95: 19.003629684448242, 96: 18.882036209106445, 97: 18.995285034179688, 98: 18.99576187133789, 99: 19.115447998046875, 100: 19.128799438476562, 101: 19.116878509521484, 102: 19.09923553466797, 103: 19.111156463623047, 104: 18.985986709594727, 105: 19.128799438476562, 106: 19.154787063598633, 107: 19.144296646118164, 108: 19.121646881103516, 109: 19.113540649414062, 110: 19.125699996948242, 111: 19.127845764160156, 112: 19.1190242767334, 113: 19.253969192504883, 114: 19.25945281982422, 115: 19.25373077392578, 116: 19.25516128540039, 117: 19.262313842773438, 118: 19.26708221435547, 119: 19.268274307250977, 120: 19.26279067993164, 121: 19.41704750061035, 122: 19.37413215637207, 123: 19.377708435058594, 124: 19.376277923583984, 125: 19.402265548706055, 126: 19.38629150390625, 127: 19.385814666748047, 128: 19.377470016479492, 129: 26.999235153198242, 130: 27.437925338745117, 131: 27.226924896240234, 132: 27.278423309326172, 133: 27.30870246887207, 134: 26.97610855102539, 135: 26.96061134338379, 136: 27.473926544189453, 137: 27.542829513549805, 138: 27.498960494995117, 139: 27.477502822875977, 140: 27.191877365112305, 141: 27.226686477661133, 142: 27.67658233642578, 143: 27.47941017150879, 144: 27.664661407470703, 145: 27.638673782348633, 146: 27.590036392211914, 147: 27.58336067199707, 148: 27.330398559570312, 149: 27.32992172241211, 150: 27.30846405029297, 151: 27.31156349182129, 152: 27.313709259033203, 153: 27.89592742919922, 154: 27.677059173583984, 155: 27.874231338500977, 156: 27.693510055541992, 157: 27.855634689331055, 158: 27.67157554626465, 159: 27.843475341796875, 160: 27.692556381225586, 161: 27.819156646728516, 162: 27.747154235839844, 163: 27.50396728515625, 164: 28.22422981262207, 165: 27.851104736328125, 166: 27.94623374938965, 167: 27.79245376586914, 168: 27.9691219329834, 169: 27.94337272644043, 170: 27.878999710083008, 171: 28.08690071105957, 172: 27.875423431396484, 173: 28.06544303894043, 174: 27.879953384399414, 175: 28.019189834594727, 176: 27.856111526489258, 177: 27.9843807220459, 178: 28.18012237548828, 179: 27.9388427734375, 180: 28.139829635620117, 181: 27.956247329711914, 182: 28.100013732910156, 183: 27.994871139526367, 184: 27.94814109802246, 185: 28.280973434448242, 186: 28.029918670654297, 187: 28.21969985961914, 188: 28.06878089904785, 189: 28.18918228149414, 190: 28.038501739501953, 191: 28.046131134033203, 192: 28.252124786376953, 193: 32.59634971618652, 194: 32.5624942779541, 195: 32.3941707611084, 196: 32.743215560913086, 197: 32.55319595336914, 198: 32.57918357849121, 199: 32.4556827545166, 200: 32.68694877624512}
        first_layer, end_layer = U.get_pp_indices(1, 1, 1)
        if first_layer == 0:
            pass
            #self.prerun_profile()
            #print(self.profile_data['decode'])

    async def step_async(
        self, virtual_engine: int, ctx_idx: int
    ) -> List[Union[RequestOutput, PoolingRequestOutput]]:
        # these are cached outputs from previous iterations. None if on first
        # iteration
        cached_outputs = None

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
            await asyncio.sleep(0.002)
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
            
            f = open('worker_trace.log', 'a')
            for scheduled_seq_group in record_seq_groups:
                request_id = scheduled_seq_group.seq_group.request_id
                req_stage = scheduled_seq_group.seq_group.seqs[0].data._stage
                if req_stage == SequenceStage.PREFILL:
                    print(f'request {request_id} got its first token at {time.time()}', file = f)

            f.close()
            
            scheduler_outputs.scheduled_seq_groups = []
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
        decode_batch_size_list = [i for i in range(1, 201)]

        sampling_params = \
                SamplingParams(top_p=0.99)
        first_layer, last_layer = U.get_pp_indices(1, 1, 1)
        last_layer -= 1
        num_layers = last_layer - first_layer
        
        # profile prefill
        print('MoLink Engine starts to profile prefill latency...')
        for batched_token_num in prefill_batched_token_list:
            seqs: List[SequenceGroupMetadata] = []
            seq_len = batched_token_num
            dummy_data = self.model_executor.driver_worker.model_runner.input_registry \
                .dummy_data_for_profiling(self.model_executor.driver_worker.model_runner.model_config,
                                            seq_len,
                                            self.model_executor.driver_worker.model_runner.mm_registry)
            seq = SequenceGroupMetadata(
                request_id=str(1),
                is_prompt=True,
                seq_data={1: dummy_data.seq_data},
                sampling_params=sampling_params,
                block_tables=None,
            )
            seqs.append(seq)
            kv_caches = [
                torch.tensor([], dtype=torch.float32, device=self.model_executor.driver_worker.model_runner.device)
                for _ in range(num_layers)
            ]
            model_input = self.model_executor.driver_worker.model_runner.prepare_model_input(seqs)

            ts = time.time()
            self.model_executor.driver_worker.model_runner.execute_model(model_input, kv_caches)
            torch.cuda.synchronize()
            te = time.time()
            # in ms
            profiled_latency = (te - ts) * 1000
            prefill_table = self.profile_data.get('prefill')
            prefill_table.update({batched_token_num : profiled_latency})

        print('Profile of prefill latency finished.')
        #print('Prefill latency stats: ')
        #for group, latency in self.profile_data['prefill'].items():
        #    print(group, latency)

        # decode profile
        print('MoLink Engine starts to profile decode latency...')
        for batch_size in decode_batch_size_list:
            seqs: List[SequenceGroupMetadata] = []
            ctn = 0
            for group_id in range(batch_size):
                seq_len = 1
                dummy_data = self.model_executor.driver_worker.model_runner.input_registry \
                    .dummy_data_for_profiling(self.model_executor.driver_worker.model_runner.model_config,
                                              seq_len,
                                              self.model_executor.driver_worker.model_runner.mm_registry)

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
                torch.tensor([], dtype=torch.float32, device=self.model_executor.driver_worker.model_runner.device)
                for _ in range(num_layers)
            ]
            model_input = self.model_executor.driver_worker.model_runner.prepare_model_input(seqs)

            ts = time.time()
            self.model_executor.driver_worker.model_runner.execute_model(model_input, kv_caches)
            torch.cuda.synchronize()
            te = time.time()
            # in ms
            profiled_latency = (te - ts) * 1000
            decode_table = self.profile_data.get('decode')
            decode_table.update({batch_size : profiled_latency})

        print('Profile of decode latency finished.')
        #print('Decode latency stats: ')
        #for group, latency in self.profile_data['decode'].items():
        #    print(group, latency)
    
    def _process_model_outputs(self,
                            ctx: SchedulerContext,
                            request_id: Optional[str] = None) -> None:
        """Apply the model output to the sequences in the scheduled seq groups
        and return responses.

        ctx: The virtual engine context to work on
        request_id: If provided, then only this request is going to be processed
        """

        now = time.time()

        if len(ctx.output_queue) == 0:
            return None

        # Get pending async postprocessor
        if request_id:
            # When we process only one request, no pop is required
            # (since later we will process all of the rest)
            (outputs, seq_group_metadata_list, scheduler_outputs, is_async,
                is_last_step, is_first_step_output, skip) = ctx.output_queue[0]
        else:
            (outputs, seq_group_metadata_list, scheduler_outputs, is_async,
                is_last_step, is_first_step_output,
                skip) = ctx.output_queue.popleft()

        # Sanity check
        assert len(seq_group_metadata_list) == len(
            scheduler_outputs.scheduled_seq_groups)

        has_multiple_outputs: bool = len(outputs) > 1
        outputs_by_sequence_group: List[List[SequenceGroupOutput]]
        if has_multiple_outputs:
            assert self.scheduler_config.is_multi_step or \
                        self.speculative_config
            # Organize outputs by [step][sequence group] instead of
            # [sequence group][step].
            if self.scheduler_config.is_multi_step:
                outputs_by_sequence_group = create_output_by_sequence_group(
                    outputs, len(seq_group_metadata_list))
            elif self.speculative_config:
                # Decodes are multi-steps while prefills are not, outputting at
                # most 1 token. Separate them so that we can trigger chunk
                # processing without having to pad or copy over prompts K times
                # to match decodes structure (costly with prompt_logprobs).
                num_prefills = sum(sg.is_prompt
                                    for sg in seq_group_metadata_list)
                prefills, decodes = outputs[:num_prefills], outputs[
                    num_prefills:]
                outputs_by_sequence_group = create_output_by_sequence_group(
                    decodes,
                    num_seq_groups=len(seq_group_metadata_list) - num_prefills)
                outputs_by_sequence_group = [p.outputs for p in prefills
                                                ] + outputs_by_sequence_group
            # We have outputs for multiple steps submitted in a single burst,
            # so invalidate is_first_step_output.
            is_first_step_output = None
        else:
            outputs_by_sequence_group = outputs

        # Determine the requests we need to operate on
        if request_id:
            indices = []
            for i, seq_group_meta in enumerate(seq_group_metadata_list):
                if seq_group_meta.request_id == request_id:
                    assert i not in skip  # Cannot be called twice
                    indices.append(i)
                    break

            # If the request_id was not found, then it means that
            # this is a new request that has no pending async
            # postprocessor
            if not indices:
                return
        else:
            indices = range(len(seq_group_metadata_list))  # type: ignore

        finished_before: List[int] = []
        finished_now: List[int] = []

        # ************************************
        f = open('worker_trace.log', 'a')
        for i in indices:
            if i in skip:
                continue

            seq_group_meta = seq_group_metadata_list[i]
            scheduled_seq_group = scheduler_outputs.scheduled_seq_groups[i]

            seq_group: SequenceGroup = scheduled_seq_group.seq_group

            if seq_group.is_finished():
                finished_before.append(i)
                continue

            output: List[SequenceGroupOutput]
            if has_multiple_outputs:
                output = outputs_by_sequence_group[i]
            else:
                output = [outputs_by_sequence_group[0][i]]

            if not is_async:
                if self.scheduler_config.is_multi_step:
                    # Updates happen only if the sequence is prefill
                    self._update_num_computed_tokens_for_multi_step_prefill(
                        seq_group, seq_group_meta, is_first_step_output)
                else:
                    seq_group.update_num_computed_tokens(
                        seq_group_meta.token_chunk_size or 0)

            if outputs:
                for o in outputs:
                    if (isinstance(o, SamplerOutput)
                            and seq_group.metrics is not None):
                        if seq_group.metrics.model_forward_time is not None:
                            seq_group.metrics.model_forward_time += (
                                o.model_forward_time or 0)
                        else:
                            seq_group.metrics.model_forward_time = (
                                o.model_forward_time)
                        if seq_group.metrics.model_execute_time is not None:
                            seq_group.metrics.model_execute_time += (
                                o.model_execute_time or 0)
                        else:
                            seq_group.metrics.model_execute_time = (
                                o.model_execute_time)

            if self.model_config.runner_type == "pooling":
                self._process_sequence_group_outputs(seq_group, output)
            else:
                self.output_processor.process_prompt_logprob(seq_group, output)
                if seq_group_meta.do_sample:
                    self.output_processor.process_outputs(
                        seq_group, output, is_async)

            if seq_group.is_finished():
                print(f'request {seq_group.request_id} finished at {time.time()}', file = f)
                print(f'request {seq_group.request_id} total token num is {len(seq_group.seqs[0].data._output_token_ids)}', file = f)
                finished_now.append(i)

        # ************************************
        f.close()

        # Generate outputs for the requests that finished this iteration
        for i in finished_now:
            scheduled_seq_group = scheduler_outputs.scheduled_seq_groups[i]

            seq_group = scheduled_seq_group.seq_group
            seq_group.maybe_set_first_token_time(now)
            if not seq_group.is_prefill():
                seq_group.set_last_token_time(now)
            request_output = RequestOutputFactory.create(
                seq_group,
                self.seq_id_to_seq_group,
                use_cache=self.use_cached_outputs)
            if request_output:
                ctx.request_outputs.append(request_output)

        # When we process a single request, we skip it for the next time,
        # and invoke the request output callback (if there was final output)
        if request_id:
            assert len(indices) == 1
            skip.append(indices[0])

            if (finished_now
                    and self.process_request_outputs_callback is not None):
                self.process_request_outputs_callback(ctx.request_outputs)
                ctx.request_outputs.clear()
            return

        # Free currently finished requests
        if finished_now:
            for scheduler in self.scheduler:
                scheduler.free_finished_seq_groups()

        # For multi-step without streaming, don't create outputs each iteration
        if not is_last_step and not ctx.multi_step_stream_outputs:
            # Immediately process request outputs here (if callback is given)
            if (finished_now
                    and self.process_request_outputs_callback is not None):
                self.process_request_outputs_callback(ctx.request_outputs)
                ctx.request_outputs.clear()
            return

        # Create the outputs
        for i in indices:
            if i in skip or i in finished_before or i in finished_now:
                continue  # Avoids double processing

            scheduled_seq_group = scheduler_outputs.scheduled_seq_groups[i]

            seq_group = scheduled_seq_group.seq_group
            seq_group.maybe_set_first_token_time(now)
            if not seq_group.is_prefill():
                seq_group.set_last_token_time(now)
            request_output = RequestOutputFactory.create(
                seq_group,
                self.seq_id_to_seq_group,
                use_cache=self.use_cached_outputs)
            if request_output:
                ctx.request_outputs.append(request_output)

        # For multi-step with streaming, create outputs each iteration
        if not is_last_step and ctx.multi_step_stream_outputs:
            # Immediately process request outputs here (if callback is given)
            if self.process_request_outputs_callback is not None:
                self.process_request_outputs_callback(ctx.request_outputs)
                ctx.request_outputs.clear()
            return

        for seq_group in scheduler_outputs.ignored_seq_groups:
            params = seq_group.sampling_params
            if params is not None and params.output_kind == (
                    RequestOutputKind.DELTA) and not seq_group.is_finished():
                continue

            request_output = RequestOutputFactory.create(
                seq_group,
                self.seq_id_to_seq_group,
                use_cache=self.use_cached_outputs,
            )
            if request_output:
                ctx.request_outputs.append(request_output)

        # Immediately process request outputs here (if callback is given)
        if (ctx.request_outputs
                and self.process_request_outputs_callback is not None):
            self.process_request_outputs_callback(ctx.request_outputs)
            ctx.request_outputs.clear()

        # For async case, we need to record the stats here.
        # For non-async case, the stats are done in the
        # LLMEngine/AsyncLLMEngine directly
        if is_async:
            # Log stats.
            self.do_log_stats(scheduler_outputs, outputs, finished_before,
                                skip)

            # Tracing
            self.do_tracing(scheduler_outputs, finished_before)

        return None


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
        self.model_hidden_size = model_config.hf_config.hidden_size
        self.model_type_size = 16

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

        # for measurement branch
        # when a worker start an engine, it deletes existing log file
        # and create a new one
        # be sure that we are at /MoLink so the log file is written to this directory
        log_file = 'worker_trace.log'
        try:
            if os.path.exists(log_file):
                os.remove(log_file)
            else:
                print(f"file does not exist: {log_file}")
            
            with open(log_file, 'w') as f:
                pass
            
            print(f"empty log file created: {log_file}")
            
        except Exception as e:
            print(f"log file creation failed: {e}")

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
                await asyncio.sleep(0.001)
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
            
            assert len(requests_in_progress) == len(has_requests_in_progress)
            if batch_num > len(requests_in_progress):
                cur_len = len(requests_in_progress)
                for i in range(cur_len, batch_num):
                    requests_in_progress.append(asyncio.create_task(engine.engine_step(0, i)))
                    has_requests_in_progress.append(True)

            for idx in range(len(requests_in_progress)):
                if idx >= batch_num:
                    has_requests_in_progress[idx] = False
                
                elif requests_in_progress[idx].done():
                    requests_in_progress[idx] = (
                        asyncio.create_task(
                            engine.engine_step(0, idx)))
                    has_requests_in_progress[idx] = True

                    if idx == 0:
                        batch_num = engine.culculate_batch_num()
            
            await asyncio.sleep(0.001)

    async def engine_step(self, virtual_engine: int, ctx_idx: int) -> bool:
        """Kick the engine to process the waiting requests.

        Returns True if there are in-progress requests."""

        new_requests, aborted_requests = (
            self._request_tracker.get_new_and_aborted_requests())

        for new_request in new_requests:
            f = open('worker_trace.log', 'a')
            print(f'request {new_request['request_id']} is added at {time.time()}', file = f)
            f.close()
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
        
    
    def culculate_compute_latency(self, num_batched_tokens, batch_size):
        #decode
        left = 1
        right = 1

        keys = sorted(self.engine.profile_data['decode'].keys())

        if batch_size == 1:
            return self.engine.profile_data['decode'].get(1)

        for i in range(0, len(keys) - 1):
            if keys[i] <= batch_size and keys[i + 1] >= batch_size:
                left = keys[i]
                right = keys[i + 1]
                break
        if left == right:
            return self.engine.profile_data['decode'].get(left)
        left_latency = self.engine.profile_data['decode'].get(left)
        right_latency = self.engine.profile_data['decode'].get(right)

        return left_latency + (right_latency - left_latency) * ((batch_size - left) / (right - left))
    

    def culculate_transmission_latency(self, num_batched_token, batch_size):
        # Mbps
        bandwidth = 1000
        # ms
        latency = 5
        # bit
        batch_data_size = self.model_type_size * self.model_hidden_size * num_batched_token * batch_size
        # ms
        return (batch_data_size) / (bandwidth * 1e6) * 1000 + latency

    def get_avg_system_overhead(self):
        # ms
        return 5
    
    def culculate_batch_num(self): 
        # equal to pipeline size
        base_batch_num = 2
        num_requests = len(self.engine.scheduler[0].waiting) + len(self.engine.scheduler[0].running)
        if num_requests <= 1:
            return 1
        if num_requests <= base_batch_num:
            return base_batch_num

        for batch_num in range(base_batch_num + 1, self.engine.max_batch_num + 1):
            schedule_limit = int(num_requests / batch_num + 1)
            self.engine.scheduler[0].set_schedule_limit(schedule_limit)
            single_batch_size = int(num_requests / batch_num)
            single_compute_latency = self.culculate_compute_latency(1, single_batch_size) + self.get_avg_system_overhead()
            single_transmission_latency = self.culculate_transmission_latency(1, single_batch_size)
            bubble = (base_batch_num) * single_transmission_latency
            if (batch_num - base_batch_num) * single_compute_latency >= bubble:
                return batch_num 
        return self.engine.max_batch_num
