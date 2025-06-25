import enum
import os
import random
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, Iterable, List, Optional
from typing import Sequence as GenericSequence
from typing import Set, Tuple, Union

from vllm.config import CacheConfig, LoRAConfig, SchedulerConfig
from vllm.core.interfaces import AllocStatus, BlockSpaceManager
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.sequence import (Sequence, SequenceData, SequenceGroup,
                           SequenceGroupBase, SequenceGroupMetadata,
                           SequenceGroupMetadataDelta, SequenceStage,
                           SequenceStatus)
from vllm.utils import Device, PyObjectCache

logger = init_logger(__name__)
from vllm.core.scheduler import Scheduler, SchedulingBudget, PartialPrefillMetadata, SchedulerRunningOutputs, ScheduledSequenceGroup, \
                                 PreemptionMode, SchedulerSwappedInOutputs, SchedulerPrefillOutputs, seq_group_metadata_builder, \
                                 scheduler_running_outputs_builder, scheduled_seq_group_builder

class MolinkScheduler(Scheduler):

    def __init__(
        self,
        scheduler_config: SchedulerConfig,
        cache_config: CacheConfig,
        lora_config: Optional[LoRAConfig],
        pipeline_parallel_size: int = 1,
        output_proc_callback: Optional[Callable] = None,
    ) -> None:
        
        super().__init__(scheduler_config, cache_config, lora_config, pipeline_parallel_size, output_proc_callback)
        # records the requests that have been scheduled by former micro batches
        self.requests_on_fly = set()
        self.schedule_limit = 10

    def set_schedule_limit(self, schedule_limit: int):
        self.schedule_limit = schedule_limit

    def _mark_seq_as_schedule_free(self, request_id: str):
        if request_id in self.requests_on_fly:
            self.requests_on_fly.remove(request_id)

    def _schedule_running(
        self,
        budget: SchedulingBudget,
        curr_loras: Optional[Set[int]],
        enable_chunking: bool = False,
        partial_prefill_metadata: Optional[PartialPrefillMetadata] = None,
    ) -> SchedulerRunningOutputs:

        try:

            ret = scheduler_running_outputs_builder()
            ret.blocks_to_swap_out.clear()
            ret.blocks_to_copy.clear()
            ret.decode_seq_groups.clear()
            ret.prefill_seq_groups.clear()
            ret.preempted.clear()
            ret.swapped_out.clear()

            ret.num_lookahead_slots = self._get_num_lookahead_slots(
                is_prefill=False, enable_chunking=enable_chunking)

            ret.decode_seq_groups_list.clear()
            ret.prefill_seq_groups_list.clear()

            # Blocks that need to be swapped or copied before model execution.
            blocks_to_swap_out: List[Tuple[int, int]] = ret.blocks_to_swap_out
            blocks_to_copy: List[Tuple[int, int]] = ret.blocks_to_copy

            decode_seq_groups: List[ScheduledSequenceGroup] = ret.decode_seq_groups
            prefill_seq_groups: List[
                ScheduledSequenceGroup] = ret.prefill_seq_groups
            preempted: List[SequenceGroup] = ret.preempted
            swapped_out: List[SequenceGroup] = ret.swapped_out

            running_queue = self.running
            assert len(self._async_stopped) == 0

            # records the seq group that should be ignore 
            reserve_queue: Deque[SequenceGroup] = deque()

            num_scheduled = 0

            while running_queue:

                if num_scheduled >= self.schedule_limit:
                    break

                seq_group = running_queue[0]
                # We discard the cached tokens info here because we don't need it
                # for running sequence:
                #   1. If a sequence is running with chunked prefill, the cached
                #      tokens info was already used for the first prefill.
                #   2. If a sequence is running with non-chunked prefill, then
                #      there it's a decoding sequence, and the cached tokens info is
                #      irrelevant.

                # judge if a seq group has already been scheduled by other batches
                request_id = seq_group.request_id
                if request_id in self.requests_on_fly:
                    running_queue.popleft()
                    reserve_queue.append(seq_group)
                    continue


                num_uncached_new_tokens, _ = \
                    self._get_num_new_uncached_and_cached_tokens(
                    seq_group,
                    SequenceStatus.RUNNING,
                    enable_chunking,
                    budget,
                    partial_prefill_metadata,
                )

                num_running_tokens = num_uncached_new_tokens
                if num_running_tokens == 0:
                    # No budget => Stop
                    break

                running_queue.popleft()

                # With async postprocessor, an extra decode run is done
                # to process the final tokens. The check below avoids this extra
                # decode run when the model max len is reached, in order to avoid
                # a memory overflow.
                if (self.use_async_output_proc and seq_group.seqs[0].get_len()
                        > self.scheduler_config.max_model_len):
                    self._async_stopped.append(seq_group)
                    continue

                # NOTE(woosuk): Preemption happens only when there is no available
                # slot to keep all the sequence groups in the RUNNING state.
                while not self._can_append_slots(seq_group, enable_chunking):
                    budget.subtract_num_batched_tokens(seq_group.request_id,
                                                    num_running_tokens)
                    num_running_seqs = seq_group.get_max_num_running_seqs()
                    budget.subtract_num_seqs(seq_group.request_id,
                                            num_running_seqs)

                    if (curr_loras is not None and seq_group.lora_int_id > 0
                            and seq_group.lora_int_id in curr_loras):
                        curr_loras.remove(seq_group.lora_int_id)

                    # Determine victim sequence
                    cont_loop = True
                    if running_queue:
                        # Preempt the lowest-priority sequence group.
                        victim_seq_group = running_queue.pop()
                    else:
                        # No other sequence group can be preempted.
                        # Preempt the current sequence group.
                        # Note: This is also where we stop this loop
                        # (since there is nothing else to preempt)
                        victim_seq_group = seq_group
                        cont_loop = False

                    # With async postprocessor, before preempting a sequence
                    # we need to ensure it has no pending async postprocessor
                    do_preempt = True
                    if self.use_async_output_proc:
                        assert self.output_proc_callback is not None
                        self.output_proc_callback(
                            request_id=victim_seq_group.request_id)

                        # It may be that the async pending "victim_seq_group"
                        # becomes finished, in which case we simply free it.
                        if victim_seq_group.is_finished():
                            self._free_finished_seq_group(victim_seq_group)
                            do_preempt = False

                    # Do preemption
                    if do_preempt:
                        preempted_mode = self._preempt(victim_seq_group,
                                                    blocks_to_swap_out)
                        if preempted_mode == PreemptionMode.RECOMPUTE:
                            preempted.append(victim_seq_group)
                        else:
                            swapped_out.append(victim_seq_group)

                    if not cont_loop:
                        break
                else:
                    self._append_slots(seq_group, blocks_to_copy, enable_chunking)

                    is_prefill = seq_group.is_prefill()
                    
                    scheduled_seq_group = scheduled_seq_group_builder()
                    scheduled_seq_group.seq_group = seq_group
                    if is_prefill:
                        scheduled_seq_group.token_chunk_size = num_running_tokens
                        prefill_seq_groups.append(scheduled_seq_group)
                        ret.prefill_seq_groups_list.append(seq_group)
                    else:
                        scheduled_seq_group.token_chunk_size = 1
                        decode_seq_groups.append(scheduled_seq_group)
                        ret.decode_seq_groups_list.append(seq_group)

                    budget.add_num_batched_tokens(seq_group.request_id,
                                                num_running_tokens)
                    # OPTIMIZATION:  Note that get_max_num_running_seqs is
                    # expensive. For the default scheduling chase where
                    # enable_chunking is False, num_seqs are updated before running
                    # this method, so we don't have to update it again here.
                    if enable_chunking:
                        num_running_seqs = seq_group.get_max_num_running_seqs()
                        budget.add_num_seqs(seq_group.request_id, num_running_seqs)
                    if curr_loras is not None and seq_group.lora_int_id > 0:
                        curr_loras.add(seq_group.lora_int_id)

                    # mark the request as on-fly
                    self.requests_on_fly.add(seq_group.request_id)
                    num_scheduled += 1
            
            # push the ignored seq groups into running queue
            while len(reserve_queue) > 0:
                sg = reserve_queue.pop()
                running_queue.appendleft(sg)

            self._scheduler_running_outputs_cache[self.next_cache_id].reset()
            self._scheduled_seq_group_cache[self.next_cache_id].reset()

            return ret
        except Exception as e:
            print(f'Exception in schedule')

    def _schedule_swapped(
        self,
        budget: SchedulingBudget,
        curr_loras: Optional[Set[int]],
        enable_chunking: bool = False,
    ) -> SchedulerSwappedInOutputs:

        # Blocks that need to be swapped or copied before model execution.
        blocks_to_swap_in: List[Tuple[int, int]] = []
        blocks_to_copy: List[Tuple[int, int]] = []
        decode_seq_groups: List[ScheduledSequenceGroup] = []
        prefill_seq_groups: List[ScheduledSequenceGroup] = []
        infeasible_seq_groups: List[SequenceGroup] = []

        swapped_queue = self.swapped

        # records the seq group that should be ignore 
        reserve_queue: Deque[SequenceGroup] = deque()

        leftover_swapped: Deque[SequenceGroup] = deque()
        while swapped_queue:
            seq_group = swapped_queue[0]

            # judge if a request has already been scheduled by other batches
            request_id = seq_group.request_id
            if request_id in self.requests_on_fly:
                swapped_queue.popleft()
                reserve_queue.append(seq_group)
                continue

            # If the sequence group cannot be swapped in, stop.
            is_prefill = seq_group.is_prefill()
            alloc_status = self.block_manager.can_swap_in(
                seq_group,
                self._get_num_lookahead_slots(is_prefill, enable_chunking))
            if alloc_status == AllocStatus.LATER:
                break
            elif alloc_status == AllocStatus.NEVER:
                logger.warning(
                    "Failing the request %s because there's not enough kv "
                    "cache blocks to run the entire sequence.",
                    seq_group.request_id,
                )
                for seq in seq_group.get_seqs():
                    seq.status = SequenceStatus.FINISHED_IGNORED
                infeasible_seq_groups.append(seq_group)
                swapped_queue.popleft()
                continue

            lora_int_id = 0
            if self.lora_enabled:
                lora_int_id = seq_group.lora_int_id
                assert curr_loras is not None
                assert self.lora_config is not None
                if (lora_int_id > 0 and (lora_int_id not in curr_loras)
                        and len(curr_loras) >= self.lora_config.max_loras):
                    # We don't have a space for another LoRA, so
                    # we ignore this request for now.
                    leftover_swapped.appendleft(seq_group)
                    swapped_queue.popleft()
                    continue

            # The total number of sequences in the RUNNING state should not
            # exceed the maximum number of sequences.
            num_new_seqs = seq_group.get_max_num_running_seqs()
            num_new_tokens_uncached, num_new_tokens_cached = (
                self._get_num_new_uncached_and_cached_tokens(
                    seq_group, SequenceStatus.SWAPPED, enable_chunking,
                    budget))

            if num_new_tokens_uncached == 0 or not budget.can_schedule(
                    num_new_tokens=num_new_tokens_uncached,
                    num_new_seqs=num_new_seqs,
            ):
                break

            if lora_int_id > 0 and curr_loras is not None:
                curr_loras.add(lora_int_id)
            swapped_queue.popleft()
            self._swap_in(seq_group, blocks_to_swap_in)
            self._append_slots(seq_group, blocks_to_copy, enable_chunking)
            if is_prefill:
                prefill_seq_groups.append(
                    ScheduledSequenceGroup(
                        seq_group,
                        token_chunk_size=num_new_tokens_uncached +
                        num_new_tokens_cached,
                    ))
            else:
                decode_seq_groups.append(
                    ScheduledSequenceGroup(seq_group, token_chunk_size=1))
            budget.add_num_batched_tokens(
                seq_group.request_id,
                num_batched_tokens=num_new_tokens_uncached,
                num_cached_tokens=num_new_tokens_cached,
            )
            budget.add_num_seqs(seq_group.request_id, num_new_seqs)

            self.requests_on_fly.add(seq_group.request_id)

        # push the ignored seq groups into swapped queue
        while len(reserve_queue) > 0:
            sg = reserve_queue.pop()
            swapped_queue.appendleft(sg)

        swapped_queue.extendleft(leftover_swapped)

        return SchedulerSwappedInOutputs(
            decode_seq_groups=decode_seq_groups,
            prefill_seq_groups=prefill_seq_groups,
            blocks_to_swap_in=blocks_to_swap_in,
            blocks_to_copy=blocks_to_copy,
            num_lookahead_slots=self._get_num_lookahead_slots(
                is_prefill=False, enable_chunking=enable_chunking),
            infeasible_seq_groups=infeasible_seq_groups,
        )

    def _schedule_prefills(
        self,
        budget: SchedulingBudget,
        curr_loras: Optional[Set[int]],
        enable_chunking: bool = False,
        partial_prefill_metadata: Optional[PartialPrefillMetadata] = None,
    ) -> SchedulerPrefillOutputs:
        if budget.remaining_token_budget() == 0:
            # Do nothing: Can't add any more prefill anyway
            return SchedulerPrefillOutputs(
                seq_groups=[],
                ignored_seq_groups=[],
                num_lookahead_slots=self._get_num_lookahead_slots(
                    is_prefill=True, enable_chunking=enable_chunking),
            )
        ignored_seq_groups: List[SequenceGroup] = []
        seq_groups: List[ScheduledSequenceGroup] = []

        waiting_queue = self.waiting

        # records the seq group that should be ignore 
        reserve_queue: Deque[SequenceGroup] = deque()

        leftover_waiting_sequences: Deque[SequenceGroup] = deque()

        num_scheduled = 0

        while self._passed_delay(time.time()) and waiting_queue:

            if num_scheduled >= self.schedule_limit:
                break

            seq_group = waiting_queue[0]

            # judge if a request has already been scheduled by other batches
            request_id = seq_group.request_id
            if request_id in self.requests_on_fly:
                waiting_queue.popleft()
                reserve_queue.append(seq_group)
                continue

            waiting_seqs = seq_group.get_seqs(status=SequenceStatus.WAITING)
            assert len(waiting_seqs) == 1, (
                "Waiting sequence group should have only one prompt "
                "sequence.")
            if (partial_prefill_metadata is not None
                    and not partial_prefill_metadata.can_schedule(seq_group)):
                leftover_waiting_sequences.appendleft(seq_group)
                waiting_queue.popleft()
                continue
            num_new_tokens_uncached, num_new_tokens_cached = (
                self._get_num_new_uncached_and_cached_tokens(
                    seq_group,
                    SequenceStatus.WAITING,
                    enable_chunking,
                    budget,
                    partial_prefill_metadata=partial_prefill_metadata,
                ))
            num_new_tokens = num_new_tokens_uncached + num_new_tokens_cached

            if not enable_chunking:
                num_prompt_tokens = waiting_seqs[0].get_len()
                assert num_new_tokens == num_prompt_tokens

            prompt_limit = self._get_prompt_limit(seq_group)
            if num_new_tokens > prompt_limit:
                logger.warning(
                    "Input prompt (%d tokens) is too long"
                    " and exceeds limit of %d",
                    num_new_tokens,
                    prompt_limit,
                )
                for seq in waiting_seqs:
                    seq.status = SequenceStatus.FINISHED_IGNORED
                ignored_seq_groups.append(seq_group)
                waiting_queue.popleft()
                continue

            num_lookahead_slots: int = 0
            if self.scheduler_config.is_multi_step and enable_chunking:
                num_lookahead_slots = self._get_num_lookahead_slots(
                    True, enable_chunking)

            # If the sequence group cannot be allocated, stop.
            can_allocate = self.block_manager.can_allocate(
                seq_group, num_lookahead_slots=num_lookahead_slots)
            if can_allocate == AllocStatus.LATER:
                break
            elif can_allocate == AllocStatus.NEVER:
                logger.warning(
                    "Input prompt (%d tokens) + lookahead slots (%d) is "
                    "too long and exceeds the capacity of block_manager",
                    num_new_tokens,
                    num_lookahead_slots,
                )
                for seq in waiting_seqs:
                    seq.status = SequenceStatus.FINISHED_IGNORED
                ignored_seq_groups.append(seq_group)
                waiting_queue.popleft()
                continue

            lora_int_id = 0
            if self.lora_enabled:
                lora_int_id = seq_group.lora_int_id
                assert curr_loras is not None
                assert self.lora_config is not None
                if (self.lora_enabled and lora_int_id > 0
                        and lora_int_id not in curr_loras
                        and len(curr_loras) >= self.lora_config.max_loras):
                    # We don't have a space for another LoRA, so
                    # we ignore this request for now.
                    leftover_waiting_sequences.appendleft(seq_group)
                    waiting_queue.popleft()
                    continue

            if (budget.num_batched_tokens
                    >= self.scheduler_config.max_num_batched_tokens):
                # We've reached the budget limit - since there might be
                # continuous prefills in the running queue, we should break
                # to avoid scheduling any new prefills.
                break

            num_new_seqs = seq_group.get_max_num_running_seqs()
            if num_new_tokens_uncached == 0 or not budget.can_schedule(
                    num_new_tokens=num_new_tokens_uncached,
                    num_new_seqs=num_new_seqs,
            ):
                break

            # Can schedule this request.
            if curr_loras is not None and lora_int_id > 0:
                curr_loras.add(lora_int_id)
            waiting_queue.popleft()
            self._allocate_and_set_running(seq_group)

            self.requests_on_fly.add(seq_group.request_id)
            num_scheduled += 1

            if partial_prefill_metadata is not None:
                partial_prefill_metadata.maybe_increment_partial_prefills(
                    seq_group)

            if enable_chunking and self.scheduler_config.is_multi_step:
                blocks_to_copy: List[Tuple[int, int]] = []
                # init_multi_step_from_lookahead_slots happens in append_slots
                self._append_slots(seq_group, blocks_to_copy, enable_chunking)
                # This assert will trip when a copy-on-write happens. This is
                # not a concern as the very first sequence-group block
                # allocation happens above. Still, we have the assert to
                # catch any edge-cases.
                assert not blocks_to_copy
            else:
                seq_group.init_multi_step_from_lookahead_slots(
                    num_lookahead_slots,
                    num_scheduler_steps=self.scheduler_config.
                    num_scheduler_steps,
                    is_multi_step=self.scheduler_config.is_multi_step,
                    enable_chunking=enable_chunking,
                )

            seq_groups.append(
                ScheduledSequenceGroup(seq_group=seq_group,
                                       token_chunk_size=num_new_tokens))
            budget.add_num_batched_tokens(
                seq_group.request_id,
                num_batched_tokens=num_new_tokens_uncached,
                num_cached_tokens=num_new_tokens_cached,
            )
            budget.add_num_seqs(seq_group.request_id, num_new_seqs)

        # push the ignored seq groups into waiting queue
        while len(reserve_queue) > 0:
            sg = reserve_queue.pop()
            waiting_queue.appendleft(sg)

        # Queue requests that couldn't be scheduled.
        waiting_queue.extendleft(leftover_waiting_sequences)
        if len(seq_groups) > 0:
            self.prev_prompt = True

        return SchedulerPrefillOutputs(
            seq_groups=seq_groups,
            ignored_seq_groups=ignored_seq_groups,
            num_lookahead_slots=self._get_num_lookahead_slots(
                is_prefill=True, enable_chunking=enable_chunking),
        )
    