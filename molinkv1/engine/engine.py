from dataclasses import fields

from vllm.logger import init_logger
from vllm.usage.usage_lib import UsageContext
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.metrics.loggers import StatLoggerFactory

from molinkv1.arg_utils import MolinkEngineArgs
from molinkv1.config import MolinkConfig, MolinkSchedulerConfig, VllmConfig1
from molinkv1.executor import MolinkExecutor

logger = init_logger(__name__)


class MolinkEngine(AsyncLLM):

    def __init__(self, *args, **kwargs) -> None:
        molink_enabled = kwargs.pop("molink_enabled", False)
        molink_initial_peer = kwargs.pop("molink_initial_peer", None)
        molink_grpc_port = kwargs.pop("molink_grpc_port", 0)
        molink_start_layer = kwargs.pop("molink_start_layer", 0)
        molink_end_layer = kwargs.pop("molink_end_layer", -1)

        config = kwargs.get("vllm_config")
        config.__class__ = VllmConfig1
        molink_config = MolinkConfig(
            enabled=molink_enabled,
            initial_peer=molink_initial_peer,
            grpc_port=molink_grpc_port,
            start_layer=molink_start_layer,
            end_layer=molink_end_layer,
        )
        config._update_attr(molink_config)

        config.parallel_config.worker_cls = "molinkv1.worker.MolinkWorker"

        self._replace_scheduler_config(config)
        self._patch_engine_core_and_init(*args, **kwargs)

    def _replace_scheduler_config(self, config):
        sched = config.scheduler_config
        if isinstance(sched, MolinkSchedulerConfig):
            return
        try:
            sched_kwargs = {
                f.name: getattr(sched, f.name)
                for f in fields(MolinkSchedulerConfig)
                if hasattr(sched, f.name)
            }
            config.scheduler_config = MolinkSchedulerConfig(**sched_kwargs)
        except Exception:
            sched.__class__ = MolinkSchedulerConfig

    def _patch_engine_core_and_init(self, *args, **kwargs):
        import vllm.v1.engine.core as engine_core_module
        from molinkv1.engine.core import MolinkEngineCoreProc

        original_cls = engine_core_module.EngineCoreProc
        original_run = original_cls.run_engine_core

        engine_core_module.EngineCoreProc.run_engine_core = (
            MolinkEngineCoreProc.run_engine_core
        )
        try:
            super().__init__(*args, **kwargs)
        finally:
            engine_core_module.EngineCoreProc.run_engine_core = original_run

    @classmethod
    def from_engine_args(
        cls,
        engine_args: MolinkEngineArgs,
        start_engine_loop: bool = True,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: list[StatLoggerFactory] | None = None,
    ) -> "AsyncLLM":
        """Create an AsyncLLM from the EngineArgs."""
        vllm_config = engine_args.create_engine_config(usage_context)
        executor_class = MolinkExecutor

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
