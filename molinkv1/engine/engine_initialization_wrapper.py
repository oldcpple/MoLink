import asyncio
import torch
import socket
import vllm.envs as envs
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.config import VllmConfig
from vllm.v1.executor import Executor
from vllm.usage.usage_lib import UsageContext
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.v1.metrics.loggers import (
    StatLoggerFactory,
    StatLoggerManager,
    load_stat_logger_plugin_factories,
)
from vllm.transformers_utils.config import maybe_register_config_serialize_by_value
from vllm.transformers_utils.tokenizer import init_tokenizer_from_configs
from vllm.logger import init_logger
from vllm.v1.engine.processor import Processor
from vllm.plugins.io_processors import get_io_processor
from vllm.v1.engine.output_processor import OutputProcessor
from vllm.tracing import init_tracer
from molinkv1.engine.core_client import DemoEngineCoreClient

logger = init_logger(__name__)

class EngineInitializationWrapper(AsyncLLM):

    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        use_cached_outputs: bool = False,
        log_requests: bool = True,
        start_engine_loop: bool = True,
        stat_loggers: list[StatLoggerFactory] | None = None,
        aggregate_engine_logging: bool = False,
        client_addresses: dict[str, str] | None = None,
        client_count: int = 1,
        client_index: int = 0,
    ) -> None:

        maybe_register_config_serialize_by_value()

        self.model_config = vllm_config.model_config
        self.vllm_config = vllm_config
        self.observability_config = vllm_config.observability_config
        self.log_requests = log_requests

        custom_stat_loggers = list(stat_loggers or [])
        custom_stat_loggers.extend(load_stat_logger_plugin_factories())

        has_custom_loggers = bool(custom_stat_loggers)
        self.log_stats = log_stats or has_custom_loggers
        if not log_stats and has_custom_loggers:
            logger.info(
                "AsyncLLM created with log_stats=False, "
                "but custom stat loggers were found; "
                "enabling logging without default stat loggers."
            )

        if self.model_config.skip_tokenizer_init:
            tokenizer = None
        else:
            tokenizer = init_tokenizer_from_configs(self.model_config)

        self.processor = Processor(self.vllm_config, tokenizer)
        self.io_processor = get_io_processor(
            self.vllm_config,
            self.model_config.io_processor_plugin,
        )

        # OutputProcessor (converts EngineCoreOutputs --> RequestOutput).
        stream_interval = self.vllm_config.scheduler_config.stream_interval
        self.output_processor = OutputProcessor(
            self.tokenizer, log_stats=self.log_stats, stream_interval=stream_interval
        )
        endpoint = self.observability_config.otlp_traces_endpoint
        if endpoint is not None:
            tracer = init_tracer("vllm.llm_engine", endpoint)
            self.output_processor.tracer = tracer

        # EngineCore (starts the engine in background process).
        self.engine_core = DemoEngineCoreClient.make_async_mp_client(
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_stats=self.log_stats,
            client_addresses=client_addresses,
            client_count=client_count,
            client_index=client_index,
        )

        # Loggers.
        self.logger_manager: StatLoggerManager | None = None
        if self.log_stats:
            self.logger_manager = StatLoggerManager(
                vllm_config=vllm_config,
                engine_idxs=self.engine_core.engine_ranks_managed,
                custom_stat_loggers=custom_stat_loggers,
                enable_default_loggers=log_stats,
                client_count=client_count,
                aggregate_engine_logging=aggregate_engine_logging,
            )
            self.logger_manager.log_engine_initialized()

        self.output_handler: asyncio.Task | None = None
        try:
            # Start output handler eagerly if we are in the asyncio eventloop.
            asyncio.get_running_loop()
            self._run_output_handler()
        except RuntimeError:
            pass

        if envs.VLLM_TORCH_PROFILER_DIR:
            logger.info(
                "Torch profiler enabled. AsyncLLM CPU traces will be collected under %s",  # noqa: E501
                envs.VLLM_TORCH_PROFILER_DIR,
            )
            worker_name = f"{socket.gethostname()}_{os.getpid()}.async_llm"
            self.profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                ],
                with_stack=envs.VLLM_TORCH_PROFILER_WITH_STACK,
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    envs.VLLM_TORCH_PROFILER_DIR, worker_name=worker_name, use_gzip=True
                ),
            )
        else:
            self.profiler = None