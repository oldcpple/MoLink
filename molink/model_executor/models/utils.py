from typing import Any, Deque, Dict, Optional, Sequence, Tuple
import torch
from vllm.model_executor.models.utils import LayerFn, PPMissingLayer, maybe_offload_to_cpu
from molink.config import MolinkConfig

def get_pp_indices(config: MolinkConfig) -> Tuple[int, int]:
    
    serving_layers = config.pipeline_config.serving_layers
    assert len(serving_layers) >= 1, 'serving layers no specified'
    start_layer = serving_layers[0]
    # to be compatible with vLLM's impl, the right side should be close
    end_layer = serving_layers[-1] + 1
    return (start_layer, end_layer)

def make_layers(
    num_hidden_layers: int,
    config: MolinkConfig,
    layer_fn: LayerFn,
    prefix: str,
) -> Tuple[int, int, torch.nn.ModuleList]:
    """Make a list of layers with the given layer function, taking
    pipeline parallelism into account.
    """
    start_layer, end_layer = get_pp_indices(config)
    modules = torch.nn.ModuleList(
        [PPMissingLayer() for _ in range(start_layer)] + [
            maybe_offload_to_cpu(layer_fn(prefix=f"{prefix}.{idx}"))
            for idx in range(start_layer, end_layer)
        ] + [PPMissingLayer() for _ in range(end_layer, num_hidden_layers)])
    return start_layer, end_layer, modules
