from vllm.config import VllmConfig
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.structured_output import StructuredOutputManager


def _molink_use_pp(vllm_config: VllmConfig) -> bool:
    molink_config = getattr(vllm_config, "molink_config", None)
    is_molink_enabled = molink_config is not None and molink_config.enabled
    return vllm_config.parallel_config.pipeline_parallel_size > 1 or is_molink_enabled


class MolinkScheduler(Scheduler):
    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        structured_output_manager: StructuredOutputManager,
        block_size: int,
        hash_block_size: int | None = None,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        include_finished_set: bool = False,
        log_stats: bool = False,
    ) -> None:
        super().__init__(
            vllm_config,
            kv_cache_config,
            structured_output_manager,
            block_size,
            hash_block_size,
            mm_registry,
            include_finished_set,
            log_stats,
        )
        self.use_pp = _molink_use_pp(vllm_config)


class MolinkAsyncScheduler(AsyncScheduler):
    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        structured_output_manager: StructuredOutputManager,
        block_size: int,
        hash_block_size: int | None = None,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        include_finished_set: bool = False,
        log_stats: bool = False,
    ) -> None:
        super().__init__(
            vllm_config,
            kv_cache_config,
            structured_output_manager,
            block_size,
            hash_block_size: int | None = None,
            mm_registry,
            include_finished_set,
            log_stats,
        )
        self.use_pp = _molink_use_pp(vllm_config)
