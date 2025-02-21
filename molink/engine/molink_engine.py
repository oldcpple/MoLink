
from typing import (Any, AsyncGenerator, Callable, Coroutine, Dict, Iterable,
                    List, Mapping, Optional, Set, Tuple, Type, Union, overload)
from vllm.config import VllmConfig
from vllm.engine.llm_engine import LLMEngine
from vllm.executor.executor_base import ExecutorBase
from vllm.engine.async_llm_engine import AsyncLLMEngine, _AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.metrics_types import StatLoggerBase
from vllm.usage.usage_lib import UsageContext
from molink.config import MolinkConfig, PipelineConfig
from molink.executor.mp_distributed_executor import MolinkMultiprocessingDistributedExecutor

class MolinkEngine(AsyncLLMEngine):

    def __init__(self, *args, **kwargs):
        config = kwargs.get('vllm_config')
        initial_peer = kwargs.get('initial_peer')
        serving_layers = kwargs.get('serving_layers')
        model_config = config.model_config
        num_all_layers = model_config.hf_config.num_hidden_layers
        layers_range = [0, num_all_layers - 1]

        if serving_layers is None or serving_layers == '' or len(serving_layers) <= 0:
            serving_layers = list(range(0, num_all_layers))
        else:
            start, end = serving_layers.split(",")
            start = int(start)
            end = int(end)
            serving_layers = list(range(start, end + 1))

        _is_first_rank = serving_layers[0] == layers_range[0]
        _is_last_rank = serving_layers[1] == layers_range[1]
        print('%' * 100)
        print(_is_first_rank, _is_last_rank)
        config.__class__ = MolinkConfig
        pipeline_config = PipelineConfig(_is_first_rank, _is_last_rank, initial_peer = initial_peer, serving_layers = serving_layers)
        config._update_attr(pipeline_config)
        kwargs['vllm_config'] = config

        self.initial_peer = initial_peer
        self.serving_layers = serving_layers
        del kwargs['initial_peer']
        del kwargs['serving_layers']

        super().__init__(*args, **kwargs)
    
    @classmethod
    def _get_executor_cls(cls,
                          engine_config: VllmConfig) -> Type[ExecutorBase]:
        return MolinkMultiprocessingDistributedExecutor
    
    @classmethod
    def from_engine_args(
        cls,
        engine_args: AsyncEngineArgs,
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
        )
        return engine
