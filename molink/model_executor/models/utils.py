from typing import Any, Deque, Dict, Optional, Sequence, Tuple
import inspect
import warnings
import torch
from torch import nn
from transformers import PretrainedConfig
from vllm.model_executor.models.utils import LayerFn, PPMissingLayer, maybe_offload_to_cpu
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.model_loader.utils import configure_quant_config
from molink.config import MolinkConfig
from molink.model_executor.model_loader.utils import get_model_architecture

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

def _initialize_model(
    vllm_config: VllmConfig,
    *,
    prefix: str = "",
) -> nn.Module:
    """Initialize a model with the given configurations."""
    model_config = vllm_config.model_config
    model_class, _ = get_model_architecture(model_config)

    if vllm_config.quant_config is not None:
        configure_quant_config(vllm_config.quant_config, model_class)

    signatures = inspect.signature(model_class.__init__)
    all_params = [param.name for param in signatures.parameters.values()]
    if "vllm_config" in all_params and "prefix" in all_params:
        # new-style model class
        with set_current_vllm_config(vllm_config, check_compile=True):
            return model_class(vllm_config=vllm_config, prefix=prefix)

    msg = ("vLLM model class should accept `vllm_config` and `prefix` as "
           "input arguments. Possibly you have an old-style model class"
           " registered from out of tree and it is used for new vLLM version. "
           "Check https://docs.vllm.ai/en/latest/design/arch_overview.html "
           "for the design and update the model class accordingly.")
    warnings.warn(msg, DeprecationWarning, stacklevel=2)
    # try to be compatible with old-style model class
    kwargs = {}
    if "prefix" in all_params:
        kwargs["prefix"] = prefix
    if "config" in all_params:
        kwargs["config"] = model_config.hf_config
    if "cache_config" in all_params:
        kwargs["cache_config"] = vllm_config.cache_config
    if "quant_config" in all_params:
        kwargs["quant_config"] = vllm_config.quant_config
    if "lora_config" in all_params:
        kwargs["lora_config"] = vllm_config.lora_config
    if "scheduler_config" in all_params:
        kwargs["scheduler_config"] = vllm_config.scheduler_config
    with set_current_vllm_config(vllm_config, check_compile=True):
        return model_class(**kwargs)

def init_vllm_registered_model(
    vllm_config: VllmConfig,
    *,
    prefix: str = "",
    hf_config: Optional[PretrainedConfig] = None,
    architectures: Optional[list[str]] = None,
) -> nn.Module:
    """
    Helper function to initialize an inner model registered to vLLM,
    based on the arguments passed to the outer vLLM model.
    """

    if hf_config is None and architectures is not None:
        # So that the architectures field is overridden
        hf_config = vllm_config.model_config.hf_config

    if hf_config is not None:
        vllm_config = vllm_config.with_hf_config(hf_config,
                                                 architectures=architectures)

    return _initialize_model(vllm_config=vllm_config, prefix=prefix)
