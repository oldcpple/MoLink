from vllm.executor.executor_base import ExecutorBase
from vllm.config import VllmConfig

class MolinkExecutorBase(ExecutorBase):
    
    def __init__(
        self,
        _is_first_rank: bool,
        _is_last_rank: bool,
        serving_blocks,
        vllm_config: VllmConfig,
    ) -> None:
        self._is_first_rank = _is_first_rank
        self._is_last_rank = _is_last_rank
        self.serving_blocks = serving_blocks
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.load_config = vllm_config.load_config
        self.parallel_config = vllm_config.parallel_config
        self.scheduler_config = vllm_config.scheduler_config
        self.device_config = vllm_config.device_config
        self.speculative_config = vllm_config.speculative_config
        self.prompt_adapter_config = vllm_config.prompt_adapter_config
        self.observability_config = vllm_config.observability_config
        self._init_executor(_is_first_rank, _is_last_rank, serving_blocks)