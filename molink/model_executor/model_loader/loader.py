import inspect
import warnings
import torch
import os
import glob
from torch import nn
from vllm.logger import init_logger
from vllm.model_executor.model_loader.utils import (ParamMapping,
                                                    set_default_torch_dtype)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizeMethodBase)
from vllm.config import (LoadConfig, LoadFormat, ModelConfig, ParallelConfig,
                         VllmConfig, set_current_vllm_config)
from vllm.model_executor.model_loader.weight_utils import (
    download_safetensors_index_file_from_hf, download_weights_from_hf,
    filter_duplicate_safetensors_files, filter_files_not_needed_for_inference,
    get_gguf_extra_tensor_names, gguf_quant_weights_iterator,
    initialize_dummy_weights, np_cache_weights_iterator, pt_weights_iterator,
    runai_safetensors_weights_iterator, safetensors_weights_iterator)
from vllm.model_executor.model_loader import (DefaultModelLoader, DummyModelLoader, TensorizerLoader, ShardedStateLoader, 
                                                     BitsAndBytesModelLoader, GGUFModelLoader, RunaiModelStreamerLoader,
                                                     BaseModelLoader)

from vllm.model_executor.model_loader.utils import device_loading_context

from .utils import get_model_architecture

logger = init_logger(__name__)

def _initialize_model(
    vllm_config: VllmConfig,
    *,
    prefix: str = "",
) -> nn.Module:
    """Initialize a model with the given configurations."""
    model_config = vllm_config.model_config
    model_class, _ = get_model_architecture(model_config)

    signatures = inspect.signature(model_class.__init__)
    all_params = [param.name for param in signatures.parameters.values()]
    if "vllm_config" in all_params and "prefix" in all_params:
        # new-style model class
        with set_current_vllm_config(vllm_config):
            return model_class(vllm_config=vllm_config, prefix=prefix)

    msg = ("vLLM model class should accept `vllm_config` and `prefix` as "
        "input arguments. Possibly you have an old-style model class"
        " registered from out of tree and it is used for new vLLM version. "
        "Check https://docs.vllm.ai/en/latest/design/arch_overview.html "
        "for the design and update the model class accordingly.")
    warnings.warn(msg, DeprecationWarning, stacklevel=2)

    logger.warning(
        "Trying to guess the arguments for old-style model class %s",
        model_class,
    )
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
    with set_current_vllm_config(vllm_config):
        return model_class(**kwargs)

class MolinkDefaultModelLoader(DefaultModelLoader):

    def load_model(self, vllm_config: VllmConfig) -> nn.Module:
        device_config = vllm_config.device_config
        model_config = vllm_config.model_config

        target_device = torch.device(device_config.device)
        with set_default_torch_dtype(model_config.dtype):
            with target_device:
                model = _initialize_model(vllm_config=vllm_config)

            weights_to_load = {name for name, _ in model.named_parameters()}
            loaded_weights = model.load_weights(
                self.get_all_weights(model_config, model))
            # We only enable strict check for non-quantized models
            # that have loaded weights tracking currently.
            if model_config.quantization is None and loaded_weights is not None:
                weights_not_loaded = weights_to_load - loaded_weights
                if weights_not_loaded:
                    raise ValueError(
                        "Following weights were not initialized from "
                        f"checkpoint: {weights_not_loaded}")

            for _, module in model.named_modules():
                quant_method = getattr(module, "quant_method", None)
                if isinstance(quant_method, QuantizeMethodBase):
                    # When quant methods need to process weights after loading
                    # (for repacking, quantizing, etc), they expect parameters
                    # to be on the global target device. This scope is for the
                    # case where cpu offloading is used, where we will move the
                    # parameters onto device for processing and back off after.
                    with device_loading_context(module, target_device):
                        quant_method.process_weights_after_loading(module)
        return model.eval()
    
class MolinkDummyModelLoader(DummyModelLoader):
    def load_model(self, vllm_config: VllmConfig) -> nn.Module:
        device_config = vllm_config.device_config
        model_config = vllm_config.model_config
        with set_default_torch_dtype(model_config.dtype):
            with torch.device(device_config.device):
                model = _initialize_model(vllm_config=vllm_config)
            # NOTE(woosuk): For accurate performance evaluation, we assign
            # random values to the weights.
            initialize_dummy_weights(model)

            for _, module in model.named_modules():
                quant_method = getattr(module, "quant_method", None)
                if quant_method is not None:
                    # When quant methods need to process weights after loading
                    # (for repacking, quantizing, etc), they expect parameters
                    # to be on the global target device. This scope is for the
                    # case where cpu offloading is used, where we will move the
                    # parameters onto device for processing and back off after.
                    with device_loading_context(
                            module, torch.device(device_config.device)):
                        quant_method.process_weights_after_loading(module)
        return model.eval()
    
class MolinkTensorizerLoader(TensorizerLoader):
    def _load_model_serialized_cpu(
        self,
        vllm_config: VllmConfig,
    ) -> nn.Module:
        """Load a serialized model with tensorizer to the CPU.

        This is only necessary when the model isn't vLLM-tensorized (see
        examples/other/tensorize_vllm_model.py) This should still
        be faster than default HuggingFace loading, but will be slower than
        loading a vLLM-tensorized model.
        """
        device_config = vllm_config.device_config
        model_config = vllm_config.model_config
        with set_default_torch_dtype(model_config.dtype):
            with torch.device(device_config.device):
                model = _initialize_model(vllm_config=vllm_config)

            model.load_weights(self._get_weights_iterator())
        return model.eval()
    
class MolinkShardedStateLoader(ShardedStateLoader):
    def load_model(self, vllm_config: VllmConfig) -> nn.Module:
        device_config = vllm_config.device_config
        model_config = vllm_config.model_config
        from safetensors.torch import safe_open

        from vllm.distributed import get_tensor_model_parallel_rank

        local_model_path = self._prepare_weights(model_config.model,
                                                 model_config.revision)

        with set_default_torch_dtype(model_config.dtype):
            with torch.device(device_config.device):
                model = _initialize_model(vllm_config=vllm_config)
                for _, module in model.named_modules():
                    quant_method = getattr(module, "quant_method", None)
                    if quant_method is not None:
                        quant_method.process_weights_after_loading(module)
            rank = get_tensor_model_parallel_rank()
            pattern = os.path.join(
                local_model_path,
                self.pattern.format(rank=rank, part="*"),
            )
            filepaths = glob.glob(pattern)
            if not filepaths:
                # TODO: support un-sharded checkpoints too
                raise ValueError(
                    f"Could not find checkpoint files '{pattern}', only "
                    f"pre-sharded checkpoints are currently supported!")
            state_dict = self._filter_subtensors(model.state_dict())
            for path in filepaths:
                with safe_open(path, framework="pt") as f:
                    for key in f.keys():  # noqa: SIM118
                        tensor = f.get_tensor(key)
                        # If loading with LoRA enabled, additional padding may
                        # be added to certain parameters. We only load into a
                        # narrowed view of the parameter data.
                        param_data = state_dict[key].data
                        param_shape = state_dict[key].shape
                        for dim, size in enumerate(tensor.shape):
                            if size < param_shape[dim]:
                                param_data = param_data.narrow(dim, 0, size)
                        if tensor.shape != param_shape:
                            logger.warning(
                                "loading tensor of shape %s into "
                                "parameter '%s' of shape %s",
                                tensor.shape,
                                key,
                                param_shape,
                            )
                        param_data.copy_(tensor)
                        state_dict.pop(key)
            if state_dict:
                raise ValueError(
                    f"Missing keys {tuple(state_dict)} in loaded state!")
        return model.eval()
    
class MolinkBitsAndBytesModelLoader(BitsAndBytesModelLoader):
    def load_model(self, vllm_config: VllmConfig) -> nn.Module:
        device_config = vllm_config.device_config
        model_config = vllm_config.model_config
        with set_default_torch_dtype(model_config.dtype):
            with torch.device(device_config.device):
                model = _initialize_model(vllm_config=vllm_config)

                self._load_weights(model_config, model)

        return model.eval()
    
class MolinkGGUFModelLoader(GGUFModelLoader):
    def load_model(self, vllm_config: VllmConfig) -> nn.Module:
        device_config = vllm_config.device_config
        model_config = vllm_config.model_config
        local_model_path = self._prepare_weights(model_config.model)
        gguf_weights_map = self._get_gguf_weights_map(model_config)
        # we can only know if tie word embeddings after mapping weights
        if "lm_head.weight" in get_gguf_extra_tensor_names(
                local_model_path, gguf_weights_map):
            model_config.hf_config.update({"tie_word_embeddings": True})

        with set_default_torch_dtype(model_config.dtype):
            with torch.device(device_config.device):
                model = _initialize_model(vllm_config=vllm_config)
            model.load_weights(
                self._get_weights_iterator(local_model_path, gguf_weights_map))
        return model
    
class MolinkRunaiModelStreamerLoader(RunaiModelStreamerLoader):
    def load_model(self, vllm_config: VllmConfig) -> nn.Module:
        """Perform streaming of the model to destination"""
        device_config = vllm_config.device_config
        model_config = vllm_config.model_config

        target_device = torch.device(device_config.device)
        with set_default_torch_dtype(model_config.dtype):
            with target_device:
                model = _initialize_model(vllm_config=vllm_config)

            model_weights = model_config.model
            if hasattr(model_config, "model_weights"):
                model_weights = model_config.model_weights
            model.load_weights(
                self._get_weights_iterator(model_weights,
                                           model_config.revision))

            for _, module in model.named_modules():
                quant_method = getattr(module, "quant_method", None)
                if quant_method is not None:
                    with device_loading_context(module, target_device):
                        quant_method.process_weights_after_loading(module)
        return model.eval()

def _initialize_model(
    vllm_config: VllmConfig,
    *,
    prefix: str = "",
) -> nn.Module:
    """Initialize a model with the given configurations."""
    model_config = vllm_config.model_config
    model_class, _ = get_model_architecture(model_config)

    signatures = inspect.signature(model_class.__init__)
    all_params = [param.name for param in signatures.parameters.values()]
    if "vllm_config" in all_params and "prefix" in all_params:
        # new-style model class
        with set_current_vllm_config(vllm_config):
            return model_class(vllm_config=vllm_config, prefix=prefix)

    msg = ("vLLM model class should accept `vllm_config` and `prefix` as "
           "input arguments. Possibly you have an old-style model class"
           " registered from out of tree and it is used for new vLLM version. "
           "Check https://docs.vllm.ai/en/latest/design/arch_overview.html "
           "for the design and update the model class accordingly.")
    warnings.warn(msg, DeprecationWarning, stacklevel=2)

    logger.warning(
        "Trying to guess the arguments for old-style model class %s",
        model_class,
    )
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
    with set_current_vllm_config(vllm_config):
        return model_class(**kwargs)
    
def get_model_loader(load_config: LoadConfig) -> BaseModelLoader:
    """Get a model loader based on the load format."""

    if isinstance(load_config.load_format, type):
        return load_config.load_format(load_config)

    if load_config.load_format == LoadFormat.DUMMY:
        return MolinkDummyModelLoader(load_config)

    if load_config.load_format == LoadFormat.TENSORIZER:
        return MolinkTensorizerLoader(load_config)

    if load_config.load_format == LoadFormat.SHARDED_STATE:
        return MolinkShardedStateLoader(load_config)

    if load_config.load_format == LoadFormat.BITSANDBYTES:
        return MolinkBitsAndBytesModelLoader(load_config)

    if load_config.load_format == LoadFormat.GGUF:
        return MolinkGGUFModelLoader(load_config)

    if load_config.load_format == LoadFormat.RUNAI_STREAMER:
        return MolinkRunaiModelStreamerLoader(load_config)

    return MolinkDefaultModelLoader(load_config)