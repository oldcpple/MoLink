from typing import (TYPE_CHECKING, Any, Dict, List, Literal, Mapping, Optional,
                    Tuple, Type, Union, cast)
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.utils import FlexibleArgumentParser, StoreBoolean
from vllm.usage.usage_lib import UsageContext
from molink.config import MolinkConfig, PipelineConfig

class MolinkEngineArgs(AsyncEngineArgs):

    initial_peer: Optional[str] = ''
    serving_layers: Optional[str] = ''

    def add_cli_args(self, parser: FlexibleArgumentParser) -> FlexibleArgumentParser:
        super().add_cli_args(parser)

        parser.add_argument(
            '--initial-peer',
            type=str,
            default='')
        
        parser.add_argument(
            '--serving-blocks',
            type=str,
            default='')

        return parser
    
    def create_engine_config(self,
                            usage_context: Optional[UsageContext] = None
                            ) -> MolinkConfig:
        config = super().create_engine_config(usage_context)
        config.__class__ = MolinkConfig
        pipeline_config = PipelineConfig(False, False, initial_peer = self.initial_peer, serving_layers = self.serving_layers)
        config._update_attr(pipeline_config)
        #config.pipeline_config.initial_peer = self.initial_peer
        #config.pipeline_config.serving_layers = self.serving_layers
        return config
