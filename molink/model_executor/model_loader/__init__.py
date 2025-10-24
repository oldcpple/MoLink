from torch import nn
from vllm.config import VllmConfig
from .loader import get_model_loader

def get_model(*, vllm_config: VllmConfig) -> nn.Module:
    loader = get_model_loader(vllm_config.load_config)
    return loader.load_model(vllm_config=vllm_config)