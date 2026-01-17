from dataclasses import asdict, fields
import time
import signal
from typing import cast, Any
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.logger import init_logger
from vllm.usage.usage_lib import UsageContext
from concurrent.futures import Future
from vllm.v1.metrics.loggers import StatLoggerFactory
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.engine.core import EngineCore, EngineCoreProc
from vllm.v1.engine import EngineCoreOutputs
from vllm.config import VllmConfig, ParallelConfig
from vllm.v1.executor import Executor
from vllm.transformers_utils.config import maybe_register_config_serialize_by_value
from vllm.utils.system_utils import decorate_logs, set_process_title

from molinkv1.executor import MolinkExecutor
from molinkv1.config import MolinkSchedulerConfig
from molinkv1.config import MolinkConfig, VllmConfig1
from molinkv1.arg_utils import MolinkEngineArgs
import queue

logger = init_logger(__name__)

class MolinkEngineCore(EngineCore):
    """
    MolinkEngineCore: Custom EngineCore implementation with micro-batch processing.
    
    This class extends vLLM's EngineCore to support MoLink's micro-batch scheduling:
    - Overrides step() to add mark_requests_schedule_free() for micro-batch support
    - Overrides step_with_batch_queue() to enable concurrent micro-batch execution
    - Tracks request scheduling state to prevent double-scheduling across batches
    
    Key features:
    - Micro-batch processing: Multiple small batches executed concurrently
    - Request state tracking: Prevents requests from being scheduled in multiple batches
    - Batch queue management: Maintains queue of executing micro-batches
    """
    
    def step(self) -> tuple[dict[int, EngineCoreOutputs], bool]:
        """Schedule, execute, and make output.

        Returns tuple of outputs and a flag indicating whether the model
        was executed.
        """

        # Check for any requests remaining in the scheduler - unfinished,
        # or finished and not yet removed from the batch.
        if not self.scheduler.has_requests():
            return {}, False
        scheduler_output = self.scheduler.schedule(virtual_engine=0)
        
        # Check if executor returns complete output (e.g., MoLink)
        returns_complete = getattr(self.model_executor, 'returns_complete_output', False)
        
        # For MoLink: if nothing to schedule, return early
        if returns_complete and scheduler_output.total_num_scheduled_tokens == 0:
            self.scheduler.mark_requests_schedule_free(scheduler_output.num_scheduled_tokens.keys())
            return {}, False
        
        future = self.model_executor.execute_model(scheduler_output, non_block=True)
        grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)
        
        with self.log_error_detail(scheduler_output):
            model_output = future.result()
            if model_output is None and not returns_complete:
                model_output = self.model_executor.sample_tokens(grammar_output)
        
        # MoLink micro-batch: mark requests as free for scheduling
        self.scheduler.mark_requests_schedule_free(scheduler_output.num_scheduled_tokens.keys())

        engine_core_outputs = self.scheduler.update_from_output(
            scheduler_output, model_output
        )

        return engine_core_outputs, scheduler_output.total_num_scheduled_tokens > 0
    
    def step_with_batch_queue(
        self,
    ) -> tuple[dict[int, EngineCoreOutputs] | None, bool]:
        """Schedule and execute batches with the batch queue.
        Note that if nothing to output in this step, None is returned.

        The execution flow is as follows:
        1. Try to schedule a new batch if the batch queue is not full.
        If a new batch is scheduled, directly return an empty engine core
        output. In other words, fulfilling the batch queue has a higher priority
        than getting model outputs.
        2. If there is no new scheduled batch, meaning that the batch queue
        is full or no other requests can be scheduled, we block until the first
        batch in the job queue is finished.
        3. Update the scheduler from the output.
        
        For MoLink micro-batch processing:
        - Each batch is assigned a virtual_engine ID (0 to max_batch_num-1)
        - Different virtual engines can execute concurrently
        - Requests are tracked to prevent double-scheduling across micro-batches
        """
        batch_queue = self.batch_queue
        assert batch_queue is not None

        # Try to schedule a new batch if the batch queue is not full, but
        # the scheduler may return an empty batch if all requests are scheduled.
        # Note that this is not blocking.
        assert len(batch_queue) < self.batch_queue_size
        
        # For MoLink micro-batch: determine virtual_engine based on queue position
        # This enables overlapping execution of different micro-batches
        virtual_engine = len(batch_queue) % self.batch_queue_size

        model_executed = False
        deferred_scheduler_output = None
        scheduled_new_batch = False  # Track if we scheduled a new batch
        
        if self.scheduler.has_requests():
            scheduler_output = self.scheduler.schedule(virtual_engine=virtual_engine)
            
            # Check if executor returns complete output (e.g., MoLink)
            returns_complete = getattr(self.model_executor, 'returns_complete_output', False)
            
            # For MoLink: if no tokens scheduled, don't execute - just wait for existing batches
            # This prevents creating empty micro-batches that could cause issues
            if returns_complete and scheduler_output.total_num_scheduled_tokens == 0:
                # Release the in-flight tracking since nothing was actually scheduled
                self.scheduler.mark_requests_schedule_free(scheduler_output.num_scheduled_tokens.keys())
                
                # If there are batches in the queue, process them; otherwise nothing to do
                if not batch_queue:
                    # No batches and nothing to schedule
                    return None, False
                # else: fall through to pop from batch_queue
            else:
                # Normal scheduling path - execute the batch
                scheduled_new_batch = True

                # **********************instrument*****************************
                if scheduler_output.total_num_scheduled_tokens > 0:
                    import os
                    server_id = os.environ.get('VLLM_SERVER_ID', '1')
                    f = open(f'server{server_id}.log', 'a')
                    cur = time.time()
                    for new_reqs in scheduler_output.scheduled_new_reqs:
                        req_id = new_reqs.req_id
                        print(f'request {req_id} is scheduled to run at {cur}', file=f)

                    for old_req_id in scheduler_output.scheduled_cached_reqs.req_ids:
                        print(f'request {old_req_id} is scheduled to run at {cur}', file=f)

                    f.close()
                # **********************instrument*****************************

                exec_future = self.model_executor.execute_model(
                    scheduler_output, non_block=True
                )
                if not self.ec_producer:
                    model_executed = scheduler_output.total_num_scheduled_tokens > 0
                
                if self.is_pooling_model or not model_executed or returns_complete:
                    # No sampling required: pooling model, no scheduled requests,
                    # or executor handles sampling internally (MoLink)
                    future = cast(Future[ModelRunnerOutput], exec_future)
                else:
                    exec_future.add_done_callback(self._log_err_callback(scheduler_output))

                    if not scheduler_output.pending_structured_output_tokens:
                        # We aren't waiting for any tokens, get any grammar output
                        # and sample immediately.
                        grammar_output = self.scheduler.get_grammar_bitmask(
                            scheduler_output
                        )
                        future = self.model_executor.sample_tokens(
                            grammar_output, non_block=True
                        )
                    else:
                        # We need to defer sampling until we have processed the model output
                        # from the prior step.
                        deferred_scheduler_output = scheduler_output

                if not deferred_scheduler_output:
                    # Add this step's future to the queue.
                    batch_queue.appendleft((future, scheduler_output))
                    if (
                        model_executed
                        and len(batch_queue) < self.batch_queue_size
                        and not batch_queue[-1][0].done()
                    ):
                        # Don't block on next worker response unless the queue is full
                        # or there are no more requests to schedule.
                        return None, True

        elif not batch_queue:
            # Queue is empty. We should not reach here since this method should
            # only be called when the scheduler contains requests or the queue
            # is non-empty.
            return None, False

        # Block until the next result is available.
        future, scheduler_output = batch_queue.pop()
        with self.log_error_detail(scheduler_output):
            model_output = future.result()

        # **********************instrument*****************************
        import os
        server_id = os.environ.get('VLLM_SERVER_ID', '1')
        f = open(f'server{server_id}.log', 'a')
        cur = time.time()
        for new_reqs in scheduler_output.scheduled_new_reqs:
            req_id = new_reqs.req_id
            print(f'request {req_id} finished an iteration at {cur}', file=f)
            print(f'request {req_id} got its first token at {cur}', file=f)

        for old_req_id in scheduler_output.scheduled_cached_reqs.req_ids:
            print(f'request {old_req_id} finished an iteration at {cur}', file=f)

        f.close()
        # **********************instrument*****************************
        
        # MoLink micro-batch: mark requests as free for scheduling
        # This allows them to be scheduled in subsequent micro-batches
        self.scheduler.mark_requests_schedule_free(scheduler_output.num_scheduled_tokens.keys())

        engine_core_outputs = self.scheduler.update_from_output(
            scheduler_output, model_output
        )

        # NOTE(nick): We can either handle the deferred tasks here or save
        # in a field and do it immediately once step_with_batch_queue is
        # re-called. The latter slightly favors TTFT over TPOT/throughput.
        if deferred_scheduler_output:
            # We now have the tokens needed to compute the bitmask for the
            # deferred request. Get the bitmask and call sample tokens.
            grammar_output = self.scheduler.get_grammar_bitmask(
                deferred_scheduler_output
            )
            future = self.model_executor.sample_tokens(grammar_output, non_block=True)
            batch_queue.appendleft((future, deferred_scheduler_output))

        return engine_core_outputs, model_executed


class MolinkEngineCoreProc(EngineCoreProc, MolinkEngineCore):
    """
    MolinkEngineCoreProc: wraps MolinkEngineCore for background process execution.
    
    This class combines EngineCoreProc's ZMQ/multiprocessing capabilities with
    MolinkEngineCore's custom scheduling logic (micro-batch processing).
    
    The MRO (Method Resolution Order) ensures:
    - EngineCoreProc's __init__ and ZMQ setup are used
    - MolinkEngineCore's step() and step_with_batch_queue() override the base methods
    """
    
    def __init__(
        self,
        vllm_config: VllmConfig,
        local_client: bool,
        handshake_address: str,
        executor_class: type[Executor],
        log_stats: bool,
        client_handshake_address: str | None = None,
        engine_index: int = 0,
    ):
        # Call EngineCoreProc's __init__ which will also call EngineCore's __init__
        # Due to MRO, this will use MolinkEngineCore's methods for step()
        super().__init__(
            vllm_config=vllm_config,
            local_client=local_client,
            handshake_address=handshake_address,
            executor_class=executor_class,
            log_stats=log_stats,
            client_handshake_address=client_handshake_address,
            engine_index=engine_index,
        )

    @staticmethod
    def run_engine_core(*args, dp_rank: int = 0, local_dp_rank: int = 0, **kwargs):
        """
        Launch MolinkEngineCore busy loop in background process.
        
        This is a custom version of EngineCoreProc.run_engine_core that instantiates
        MolinkEngineCoreProc instead of EngineCoreProc.
        """
        # Signal handler used for graceful termination.
        shutdown_requested = False

        # Ensure we can serialize transformer config after spawning
        maybe_register_config_serialize_by_value()

        def signal_handler(signum, frame):
            nonlocal shutdown_requested
            if not shutdown_requested:
                shutdown_requested = True
                raise SystemExit()

        # Either SIGTERM or SIGINT will terminate the engine_core
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        engine_core: MolinkEngineCoreProc | None = None
        try:
            parallel_config: ParallelConfig = kwargs["vllm_config"].parallel_config
            if parallel_config.data_parallel_size > 1 or dp_rank > 0:
                set_process_title("MolinkEngineCore", f"DP{dp_rank}")
                decorate_logs()
                # Set data parallel rank for this engine process.
                parallel_config.data_parallel_rank = dp_rank
                parallel_config.data_parallel_rank_local = local_dp_rank
                # For DP mode, we still use MolinkEngineCoreProc
                # (not creating a separate DPMolinkEngineCoreProc for now)
                engine_core = MolinkEngineCoreProc(*args, **kwargs)
            else:
                set_process_title("MolinkEngineCore")
                decorate_logs()
                engine_core = MolinkEngineCoreProc(*args, **kwargs)

            logger.info("MolinkEngineCore started successfully with micro-batch support")
            engine_core.run_busy_loop()

        except SystemExit:
            logger.debug("MolinkEngineCore exiting.")
            raise
        except Exception as e:
            logger.exception("Fatal error in MolinkEngineCore: %s", e)
            raise
        finally:
            if engine_core is not None:
                engine_core.shutdown()


class MolinkEngine(AsyncLLM):

    def __init__(self, *args, **kwargs) -> None:
        # Monkey-patch vllm to use MolinkEngineCoreProc.run_engine_core
        # This ensures that when vllm creates engine core processes, it uses
        # our custom MolinkEngineCore implementation
        import vllm.v1.engine.utils as engine_utils
        # Store original to restore later if needed
        if not hasattr(engine_utils, '_original_run_engine_core'):
            from vllm.v1.engine.core import EngineCoreProc as OriginalEngineCoreProc
            engine_utils._original_run_engine_core = OriginalEngineCoreProc.run_engine_core
        
        # Replace with our custom run_engine_core
        # We need to monkey-patch the module to inject MolinkEngineCoreProc
        import vllm.v1.engine.core as core_module
        # Save original class
        if not hasattr(core_module, '_OriginalEngineCoreProc'):
            core_module._OriginalEngineCoreProc = core_module.EngineCoreProc
        # Replace with MolinkEngineCoreProc
        core_module.EngineCoreProc = MolinkEngineCoreProc
        
        config = kwargs.get("vllm_config")
        molink_enabled = kwargs.get("molink_enabled")
        molink_initial_peer = kwargs.get("molink_initial_peer")
        molink_grpc_port = kwargs.get("molink_grpc_port")
        molink_start_layer = kwargs.get("molink_start_layer")
        molink_end_layer = kwargs.get("molink_end_layer")
        config.__class__ = VllmConfig1
        molink_config = MolinkConfig(
            molink_enabled,
            molink_initial_peer,
            molink_grpc_port,
            molink_start_layer,
            molink_end_layer,
        )
        config._update_attr(molink_config)

        # Set worker class to MolinkWorker
        config.parallel_config.worker_cls = "molinkv1.worker.MolinkWorker"

        # Replace scheduler config with MolinkSchedulerConfig, preserving fields
        try:
            sched = config.scheduler_config
            if not isinstance(sched, MolinkSchedulerConfig):
                try:
                    sched_kwargs = asdict(sched)
                except TypeError:
                    # Fallback: copy overlapping fields
                    sched_kwargs = {
                        f.name: getattr(sched, f.name)
                        for f in fields(MolinkSchedulerConfig)
                        if hasattr(sched, f.name)
                    }
                config.scheduler_config = MolinkSchedulerConfig(**sched_kwargs)
        except Exception:
            # As a last resort, coerce the class
            try:
                config.scheduler_config.__class__ = MolinkSchedulerConfig
            except Exception:
                pass

        del kwargs["molink_enabled"]
        del kwargs["molink_initial_peer"]
        del kwargs["molink_grpc_port"]
        del kwargs["molink_start_layer"]
        del kwargs["molink_end_layer"]
        super().__init__(*args, **kwargs)

    
    @classmethod
    def from_vllm_config(
        cls,
        vllm_config: VllmConfig,
        start_engine_loop: bool = True,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: list[StatLoggerFactory] | None = None,
        enable_log_requests: bool = False,
        aggregate_engine_logging: bool = False,
        disable_log_stats: bool = False,
        client_addresses: dict[str, str] | None = None,
        client_count: int = 1,
        client_index: int = 0,
        disable_log_requests: bool = True,  # Deprecated, will be removed
        molink_enabled: bool = False,
        molink_initial_peer: str | None = None,
        molink_grpc_port: int = 0,
        molink_start_layer: int = 0,
        molink_end_layer: int = -1,
        **kwargs,
    ) -> "MolinkEngine":
        """Create a MolinkEngine from VllmConfig.
        
        This override accepts MoLink-specific parameters in addition to
        standard AsyncLLM parameters.
        """
        executor_class = MolinkExecutor
        return cls(
            vllm_config=vllm_config,
            executor_class=executor_class,
            start_engine_loop=start_engine_loop,
            stat_loggers=stat_loggers,
            log_requests=enable_log_requests,
            log_stats=not disable_log_stats,
            aggregate_engine_logging=aggregate_engine_logging,
            usage_context=usage_context,
            client_addresses=client_addresses,
            client_count=client_count,
            client_index=client_index,
            molink_enabled=molink_enabled,
            molink_initial_peer=molink_initial_peer,
            molink_grpc_port=molink_grpc_port,
            molink_start_layer=molink_start_layer,
            molink_end_layer=molink_end_layer,
            **kwargs,
        )

    @classmethod
    def from_engine_args(
        cls,
        engine_args: MolinkEngineArgs,
        start_engine_loop: bool = True,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: list[StatLoggerFactory] | None = None,
    ) -> "AsyncLLM":
        """Create an AsyncLLM from the EngineArgs."""
        # Create the engine configs.
        vllm_config = engine_args.create_engine_config(usage_context)
        executor_class = MolinkExecutor
        # Create the AsyncLLM.
        return cls(
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_requests=engine_args.enable_log_requests,
            log_stats=not engine_args.disable_log_stats,
            start_engine_loop=start_engine_loop,
            usage_context=usage_context,
            stat_loggers=stat_loggers,
            molink_enabled=engine_args.molink_enabled,
            molink_initial_peer=engine_args.molink_initial_peer,
            molink_grpc_port=engine_args.molink_grpc_port,
            molink_start_layer=engine_args.molink_start_layer,
            molink_end_layer=engine_args.molink_end_layer,
        )
