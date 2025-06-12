
from typing import (Any, AsyncGenerator, Callable, Coroutine, Dict, Iterable,
                    List, Mapping, Optional, Set, Tuple, Type, Union, overload)
from vllm.config import VllmConfig
from vllm.engine.llm_engine import SchedulerOutputState
from vllm.executor.executor_base import ExecutorBase
from vllm.engine.async_llm_engine import AsyncLLMEngine, _AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.metrics_types import StatLoggerBase
from vllm.usage.usage_lib import UsageContext
from vllm.outputs import PoolingRequestOutput, RequestOutput
from vllm.sequence import ExecuteModelRequest
from molink.config import MolinkConfig, PipelineConfig
from molink.executor.mp_distributed_executor import MolinkMultiprocessingDistributedExecutor
from .arg_utils import MolinkEngineArgs
import molink.distributed.parallel_state as P
import vllm.distributed.utils as U
import time
class _MolinkEngine(_AsyncLLMEngine):

    async def step_async(
        self, virtual_engine: int
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

        cached_outputs = self.cached_scheduler_outputs[virtual_engine]
        seq_group_metadata_list = cached_outputs.seq_group_metadata_list
        scheduler_outputs = cached_outputs.scheduler_outputs
        allow_async_output_proc = cached_outputs.allow_async_output_proc

        ctx = self.scheduler_contexts[virtual_engine]

        # Clear outputs for each new scheduler iteration
        ctx.request_outputs.clear()

        # skip the scheduler if there are any remaining steps in the seq groups.
        # This ensures that the scheduler is only called again when the current
        # batch has completed.
        if not self._has_remaining_steps(seq_group_metadata_list):

            # Schedule iteration
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
                    virtual_engine, seq_group_metadata_list, scheduler_outputs,
                    allow_async_output_proc)
        else:
            finished_requests_ids = list()

        assert seq_group_metadata_list is not None
        assert scheduler_outputs is not None

        if not scheduler_outputs.is_empty():

            # Check if we have a cached last_output from the previous iteration.
            # For supporting PP this is probably the best way to pass the
            # sampled_token_ids, as a separate broadcast over all the PP stages
            # will cause one virtual engine's microbatch to block the pipeline.
            last_sampled_token_ids = \
                self._get_last_sampled_token_ids(virtual_engine)

            execute_model_req = ExecuteModelRequest(
                seq_group_metadata_list=seq_group_metadata_list,
                blocks_to_swap_in=scheduler_outputs.blocks_to_swap_in,
                blocks_to_swap_out=scheduler_outputs.blocks_to_swap_out,
                blocks_to_copy=scheduler_outputs.blocks_to_copy,
                virtual_engine=virtual_engine,
                num_lookahead_slots=scheduler_outputs.num_lookahead_slots,
                running_queue_size=scheduler_outputs.running_queue_size,
                finished_requests_ids=finished_requests_ids,
                # We use ExecuteModelRequest to pass the last sampled_token_ids
                # to each of the non-last PP stages for in-place prepare_input.
                last_sampled_token_ids=last_sampled_token_ids)

            if allow_async_output_proc:
                execute_model_req.async_callback = self.async_callbacks[
                    virtual_engine]
                
            # Execute the model.
            outputs = await self.model_executor.execute_model_async(
                execute_model_req)
            
            # we set it to None during execution
            if allow_async_output_proc:
                execute_model_req.async_callback = self.async_callbacks[
                        virtual_engine]
                execute_model_req.async_callback()

            # we need to do this here so that last step's sampled_token_ids can
            # be passed to the next iteration for PP.
            if self.scheduler_config.is_multi_step:
                self._update_cached_scheduler_output(virtual_engine, outputs)
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
                    virtual_engine] = SchedulerOutputState()

            # is_first_step_output is True only when the num_steps of all
            # the sequences are 1. When the num_steps > 1,
            # multi_step_model_runner does the first-step output append.
            is_first_step_output: bool = False if not seq_group_metadata_list \
                else seq_group_metadata_list[0].state.num_steps == 1

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
            # Multi-step case
            return ctx.request_outputs

        if not self.has_unfinished_requests():
            # Drain async postprocessor (if exists)
            if len(ctx.output_queue) > 0:
                self._process_model_outputs(ctx=ctx)
            assert len(ctx.output_queue) == 0

        return ctx.request_outputs

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
        
        patch_get_pp_indices(serving_layers[0], serving_layers[1] + 1)

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


def patch_get_pp_indices(start: int, end: int):
    import vllm
    import os
    import re

    vllm_path = os.path.dirname(vllm.__file__)
    utils_path = os.path.join(vllm_path, 'distributed', 'utils.py')

    with open(utils_path, 'r') as f:
        lines = f.readlines()

    start_idx = None
    for i, line in enumerate(lines):
        if re.match(r"\s*def\s+get_pp_indices\s*\(", line):
            start_idx = i
            break

    if start_idx is None:
        print("get_pp_indices not found in utils.py")
        return

    indent = re.match(r"(\s*)def", lines[start_idx]).group(1)

    end_idx = start_idx + 1
    while end_idx < len(lines):
        line = lines[end_idx]
        if line.strip() == "":
            end_idx += 1
            continue
        if not line.startswith(indent + " "): 
            break
        end_idx += 1

    new_func = [
        f"{indent}def get_pp_indices(a, b, c):\n",
        f"{indent}    return ({start}, {end})\n"
    ]

    lines[start_idx:end_idx] = new_func

    with open(utils_path, 'w') as f:
        f.writelines(lines)

    print(f"get_pp_indices patched to return ({start}, {end}) in {utils_path}")
