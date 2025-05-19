import hashlib
from dataclasses import dataclass, field, replace
from pydantic import BaseModel, Field, PrivateAttr
from typing import (TYPE_CHECKING, Any, Callable, ClassVar, Counter, Dict,
                    Final, List, Literal, Mapping, Optional, Protocol, Set,
                    Tuple, Type, Union)
from vllm.config import VllmConfig, ModelConfig
from vllm.model_executor.models import ModelRegistry
from vllm.logger import init_logger

logger = init_logger(__name__)

class PipelineConfig():

    def __init__(self, _is_first_rank: Optional[bool], _is_last_rank: Optional[bool], initial_peer, serving_layers):
        self._is_first_rank = _is_first_rank
        self._is_last_rank = _is_last_rank
        self.initial_peer = initial_peer
        self.serving_layers = serving_layers

class MoLinkModelConfig(ModelConfig):

    def verify_with_parallel_config(
        self,
        parallel_config: "ParallelConfig",
    ) -> None:
        total_num_attention_heads = getattr(self.hf_text_config,
                                            "num_attention_heads", 0)
        tensor_parallel_size = parallel_config.tensor_parallel_size
        if total_num_attention_heads % tensor_parallel_size != 0:
            raise ValueError(
                f"Total number of attention heads ({total_num_attention_heads})"
                " must be divisible by tensor parallel size "
                f"({tensor_parallel_size}).")

        pipeline_parallel_size = parallel_config.pipeline_parallel_size
        if pipeline_parallel_size > 1:
            architectures = getattr(self.hf_config, "architectures", [])
            if not ModelRegistry.is_pp_supported_model(architectures):
                if "Qwen3ForCausalLM"  in architectures or "Qwen3MoeForCausalLM" in architectures:
                    pass
                else:
                    raise NotImplementedError(
                    "Pipeline parallelism is not supported for this model. "
                    "Supported models implement the `SupportsPP` interface.")

            if self.use_async_output_proc:
                logger.warning("Async output processor is not supported with "
                               "pipeline parallelism currently. Disabling it.")
                self.use_async_output_proc = False


@dataclass
class MolinkConfig(VllmConfig):

    pipeline_config: PipelineConfig = field(default_factory=PipelineConfig, init=True)
    model_config: MoLinkModelConfig = field(default=None, init=True)

    def _update_attr(self, pipeline_config: PipelineConfig):
        self.pipeline_config = pipeline_config