
from typing import (Any, AsyncGenerator, Callable, Coroutine, Dict, Iterable,
                    List, Mapping, Optional, Set, Tuple, Type, Union, overload)
from vllm.config import VllmConfig
from vllm.engine.llm_engine import LLMEngine
from vllm.executor.executor_base import ExecutorBase
from vllm.engine.async_llm_engine import AsyncLLMEngine, _AsyncLLMEngine
from molink.executor.mp_distributed_executor import MolinkMultiprocessingDistributedExecutor

class MolinkEngine(AsyncLLMEngine):

    def __init__(self, *args, **kwargs):
        vllm_config = kwargs.get('vllm_config')
        pipeline_config = vllm_config.pipeline_config
        initial_peer = pipeline_config.initial_peer
        serving_layers = pipeline_config.serving_layers
        model_config = vllm_config.model_config
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
        vllm_config.pipeline_config._is_first_rank = _is_first_rank
        vllm_config.pipeline_config._is_last_rank = _is_last_rank
        kwargs['vllm_config'] = vllm_config

        self.initial_peer = initial_peer
        self.serving_layers = serving_layers


        super().__init__(*args, **kwargs)
    
    @classmethod
    def _get_executor_cls(cls,
                          engine_config: VllmConfig) -> Type[ExecutorBase]:
        return MolinkMultiprocessingDistributedExecutor
    
