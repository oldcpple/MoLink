import hashlib
from dataclasses import dataclass, field, replace
from pydantic import BaseModel, Field, PrivateAttr
from typing import (TYPE_CHECKING, Any, Callable, ClassVar, Counter, Dict,
                    Final, List, Literal, Mapping, Optional, Protocol, Set,
                    Tuple, Type, Union)
from vllm.config import VllmConfig

class PipelineConfig():

    def __init__(self, _is_first_rank: Optional[bool], _is_last_rank: Optional[bool], initial_peer, serving_layers):
        self._is_first_rank = _is_first_rank
        self._is_last_rank = _is_last_rank
        self.initial_peer = initial_peer
        self.serving_layers = serving_layers

@dataclass
class MolinkConfig(VllmConfig):

    pipeline_config: PipelineConfig = field(default_factory=PipelineConfig, init=True)

    def _update_attr(self, pipeline_config: PipelineConfig):
        self.pipeline_config = pipeline_config