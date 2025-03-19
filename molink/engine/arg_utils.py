from typing import (TYPE_CHECKING, Any, Dict, List, Literal, Mapping, Optional,
                    Tuple, Type, Union, cast)
import argparse
import dataclasses
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
            '--serving-layers',
            type=str,
            default='')
        
        parser.add_argument(
            '--use-dht',
            type=bool,
            default=False)
        
        return parser
    
    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        # Get the list of attributes of this dataclass.
        attrs = [attr.name for attr in dataclasses.fields(cls)]
        # Set the attributes from the parsed arguments.
        engine_args = cls(**{attr: getattr(args, attr) for attr in attrs})
        engine_args.initial_peer = args.initial_peer
        engine_args.serving_layers = args.serving_layers
        engine_args.use_dht = args.use_dht
        engine_args.port = args.port
        return engine_args
    
    def create_engine_config(self,
                            usage_context: Optional[UsageContext] = None
                            ) -> MolinkConfig:
        config = super().create_engine_config(usage_context)
        return config
