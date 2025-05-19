from typing import (TYPE_CHECKING, Any, Dict, List, Literal, Mapping, Optional,
                    Tuple, Type, Union, cast)
import argparse
import dataclasses
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.utils import FlexibleArgumentParser, StoreBoolean
from vllm.usage.usage_lib import UsageContext
import vllm.envs as envs
from vllm.transformers_utils.utils import check_gguf_file
from vllm.config import (CacheConfig, CompilationConfig, ConfigFormat,
                         DecodingConfig, DeviceConfig, HfOverrides,
                         KVTransferConfig, LoadConfig, LoadFormat, LoRAConfig,
                         ModelConfig, ModelImpl, ObservabilityConfig,
                         ParallelConfig, PoolerConfig, PromptAdapterConfig,
                         SchedulerConfig, SpeculativeConfig, TaskOption,
                         TokenizerPoolConfig, VllmConfig)
from vllm.logger import init_logger
from molink.config import MolinkConfig, PipelineConfig, MoLinkModelConfig

logger = init_logger(__name__)

ALLOWED_DETAILED_TRACE_MODULES = ["model", "worker", "all"]

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
        
        parser.add_argument(
            '--in-autodl',
            type=bool,
            default=False)
        
        parser.add_argument(
            '--autodl-worker-num',
            type=int,
            default=0)

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
        engine_args.in_autodl = args.in_autodl
        engine_args.autodl_worker_num = args.autodl_worker_num
        return engine_args
    
    def create_model_config(self) -> MoLinkModelConfig:
        return MoLinkModelConfig(
            model=self.model,
            task=self.task,
            # We know this is not None because we set it in __post_init__
            tokenizer=cast(str, self.tokenizer),
            tokenizer_mode=self.tokenizer_mode,
            trust_remote_code=self.trust_remote_code,
            allowed_local_media_path=self.allowed_local_media_path,
            dtype=self.dtype,
            seed=self.seed,
            revision=self.revision,
            code_revision=self.code_revision,
            rope_scaling=self.rope_scaling,
            rope_theta=self.rope_theta,
            hf_overrides=self.hf_overrides,
            tokenizer_revision=self.tokenizer_revision,
            max_model_len=self.max_model_len,
            quantization=self.quantization,
            enforce_eager=self.enforce_eager,
            max_seq_len_to_capture=self.max_seq_len_to_capture,
            max_logprobs=self.max_logprobs,
            disable_sliding_window=self.disable_sliding_window,
            skip_tokenizer_init=self.skip_tokenizer_init,
            served_model_name=self.served_model_name,
            limit_mm_per_prompt=self.limit_mm_per_prompt,
            use_async_output_proc=not self.disable_async_output_proc,
            config_format=self.config_format,
            mm_processor_kwargs=self.mm_processor_kwargs,
            disable_mm_preprocessor_cache=self.disable_mm_preprocessor_cache,
            override_neuron_config=self.override_neuron_config,
            override_pooler_config=self.override_pooler_config,
            logits_processor_pattern=self.logits_processor_pattern,
            generation_config=self.generation_config,
            override_generation_config=self.override_generation_config,
            enable_sleep_mode=self.enable_sleep_mode,
            model_impl=self.model_impl,
        )

    def create_engine_config(self,
                             usage_context: Optional[UsageContext] = None
                             ) -> VllmConfig:
        if envs.VLLM_USE_V1:
            self._override_v1_engine_args(usage_context)

        # gguf file needs a specific model loader and doesn't use hf_repo
        if check_gguf_file(self.model):
            self.quantization = self.load_format = "gguf"

        # bitsandbytes quantization needs a specific model loader
        # so we make sure the quant method and the load format are consistent
        if (self.quantization == "bitsandbytes" or
           self.qlora_adapter_name_or_path is not None) and \
           self.load_format != "bitsandbytes":
            raise ValueError(
                "BitsAndBytes quantization and QLoRA adapter only support "
                f"'bitsandbytes' load format, but got {self.load_format}")

        if (self.load_format == "bitsandbytes" or
            self.qlora_adapter_name_or_path is not None) and \
            self.quantization != "bitsandbytes":
            raise ValueError(
                "BitsAndBytes load format and QLoRA adapter only support "
                f"'bitsandbytes' quantization, but got {self.quantization}")

        assert self.cpu_offload_gb >= 0, (
            "CPU offload space must be non-negative"
            f", but got {self.cpu_offload_gb}")

        device_config = DeviceConfig(device=self.device)
        model_config = self.create_model_config()

        if (model_config.is_multimodal_model and not envs.VLLM_USE_V1
                and self.enable_prefix_caching):
            logger.warning("--enable-prefix-caching is currently not "
                           "supported for multimodal models in v0 and "
                           "has been disabled.")
            self.enable_prefix_caching = False

        cache_config = CacheConfig(
            block_size=self.block_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            swap_space=self.swap_space,
            cache_dtype=self.kv_cache_dtype,
            is_attention_free=model_config.is_attention_free,
            num_gpu_blocks_override=self.num_gpu_blocks_override,
            sliding_window=model_config.get_sliding_window(),
            enable_prefix_caching=self.enable_prefix_caching,
            cpu_offload_gb=self.cpu_offload_gb,
            calculate_kv_scales=self.calculate_kv_scales,
        )
        parallel_config = ParallelConfig(
            pipeline_parallel_size=self.pipeline_parallel_size,
            tensor_parallel_size=self.tensor_parallel_size,
            max_parallel_loading_workers=self.max_parallel_loading_workers,
            disable_custom_all_reduce=self.disable_custom_all_reduce,
            tokenizer_pool_config=TokenizerPoolConfig.create_config(
                self.tokenizer_pool_size,
                self.tokenizer_pool_type,
                self.tokenizer_pool_extra_config,
            ),
            ray_workers_use_nsight=self.ray_workers_use_nsight,
            distributed_executor_backend=self.distributed_executor_backend,
            worker_cls=self.worker_cls,
        )

        max_model_len = model_config.max_model_len
        use_long_context = max_model_len > 32768
        if self.enable_chunked_prefill is None:
            # If not explicitly set, enable chunked prefill by default for
            # long context (> 32K) models. This is to avoid OOM errors in the
            # initial memory profiling phase.

            # For multimodal models, chunked prefill is disabled by default in
            # V0, but enabled by design in V1
            if model_config.is_multimodal_model:
                self.enable_chunked_prefill = bool(envs.VLLM_USE_V1)

            elif use_long_context:
                is_gpu = device_config.device_type == "cuda"
                use_sliding_window = (model_config.get_sliding_window()
                                      is not None)
                use_spec_decode = self.speculative_model is not None
                from vllm.platforms import current_platform
                if (is_gpu and not use_sliding_window and not use_spec_decode
                        and not self.enable_lora
                        and not self.enable_prompt_adapter
                        and model_config.runner_type != "pooling"
                        and not current_platform.is_rocm()):
                    self.enable_chunked_prefill = True
                    logger.warning(
                        "Chunked prefill is enabled by default for models with "
                        "max_model_len > 32K. Currently, chunked prefill might "
                        "not work with some features or models. If you "
                        "encounter any issues, please disable chunked prefill "
                        "by setting --enable-chunked-prefill=False.")
            if self.enable_chunked_prefill is None:
                self.enable_chunked_prefill = False

        if not self.enable_chunked_prefill and use_long_context:
            logger.warning(
                "The model has a long context length (%s). This may cause OOM "
                "errors during the initial memory profiling phase, or result "
                "in low performance due to small KV cache space. Consider "
                "setting --max-model-len to a smaller value.", max_model_len)
        elif (self.enable_chunked_prefill
              and model_config.runner_type == "pooling"):
            msg = "Chunked prefill is not supported for pooling models"
            raise ValueError(msg)


        speculative_config = SpeculativeConfig.maybe_create_spec_config(
            target_model_config=model_config,
            target_parallel_config=parallel_config,
            target_dtype=self.dtype,
            speculative_model=self.speculative_model,
            speculative_model_quantization = \
                self.speculative_model_quantization,
            speculative_draft_tensor_parallel_size = \
                self.speculative_draft_tensor_parallel_size,
            num_speculative_tokens=self.num_speculative_tokens,
            speculative_disable_mqa_scorer=self.speculative_disable_mqa_scorer,
            speculative_disable_by_batch_size=self.
            speculative_disable_by_batch_size,
            speculative_max_model_len=self.speculative_max_model_len,
            enable_chunked_prefill=self.enable_chunked_prefill,
            disable_log_stats=self.disable_log_stats,
            ngram_prompt_lookup_max=self.ngram_prompt_lookup_max,
            ngram_prompt_lookup_min=self.ngram_prompt_lookup_min,
            draft_token_acceptance_method=\
                self.spec_decoding_acceptance_method,
            typical_acceptance_sampler_posterior_threshold=self.
            typical_acceptance_sampler_posterior_threshold,
            typical_acceptance_sampler_posterior_alpha=self.
            typical_acceptance_sampler_posterior_alpha,
            disable_logprobs=self.disable_logprobs_during_spec_decoding,
        )

        # Reminder: Please update docs/source/features/compatibility_matrix.md
        # If the feature combo become valid
        if self.num_scheduler_steps > 1:
            if speculative_config is not None:
                raise ValueError("Speculative decoding is not supported with "
                                 "multi-step (--num-scheduler-steps > 1)")
            if self.enable_chunked_prefill and self.pipeline_parallel_size > 1:
                raise ValueError("Multi-Step Chunked-Prefill is not supported "
                                 "for pipeline-parallel-size > 1")
            from vllm.platforms import current_platform
            if current_platform.is_cpu():
                logger.warning("Multi-Step (--num-scheduler-steps > 1) is "
                               "currently not supported for CPUs and has been "
                               "disabled.")
                self.num_scheduler_steps = 1

        # make sure num_lookahead_slots is set the higher value depending on
        # if we are using speculative decoding or multi-step
        num_lookahead_slots = max(self.num_lookahead_slots,
                                  self.num_scheduler_steps - 1)
        num_lookahead_slots = num_lookahead_slots \
            if speculative_config is None \
            else speculative_config.num_lookahead_slots

        if not self.use_v2_block_manager:
            logger.warning(
                "[DEPRECATED] Block manager v1 has been removed, "
                "and setting --use-v2-block-manager to True or False has "
                "no effect on vLLM behavior. Please remove "
                "--use-v2-block-manager in your engine argument. "
                "If your use case is not supported by "
                "SelfAttnBlockSpaceManager (i.e. block manager v2),"
                " please file an issue with detailed information.")

        scheduler_config = SchedulerConfig(
            runner_type=model_config.runner_type,
            max_num_batched_tokens=self.max_num_batched_tokens,
            max_num_seqs=self.max_num_seqs,
            max_model_len=model_config.max_model_len,
            num_lookahead_slots=num_lookahead_slots,
            delay_factor=self.scheduler_delay_factor,
            enable_chunked_prefill=self.enable_chunked_prefill,
            is_multimodal_model=model_config.is_multimodal_model,
            preemption_mode=self.preemption_mode,
            num_scheduler_steps=self.num_scheduler_steps,
            multi_step_stream_outputs=self.multi_step_stream_outputs,
            send_delta_data=(envs.VLLM_USE_RAY_SPMD_WORKER
                             and parallel_config.use_ray),
            policy=self.scheduling_policy)
        lora_config = LoRAConfig(
            bias_enabled=self.enable_lora_bias,
            max_lora_rank=self.max_lora_rank,
            max_loras=self.max_loras,
            fully_sharded_loras=self.fully_sharded_loras,
            lora_extra_vocab_size=self.lora_extra_vocab_size,
            long_lora_scaling_factors=self.long_lora_scaling_factors,
            lora_dtype=self.lora_dtype,
            max_cpu_loras=self.max_cpu_loras if self.max_cpu_loras
            and self.max_cpu_loras > 0 else None) if self.enable_lora else None

        if self.qlora_adapter_name_or_path is not None and \
            self.qlora_adapter_name_or_path != "":
            if self.model_loader_extra_config is None:
                self.model_loader_extra_config = {}
            self.model_loader_extra_config[
                "qlora_adapter_name_or_path"] = self.qlora_adapter_name_or_path

        load_config = self.create_load_config()

        prompt_adapter_config = PromptAdapterConfig(
            max_prompt_adapters=self.max_prompt_adapters,
            max_prompt_adapter_token=self.max_prompt_adapter_token) \
                                        if self.enable_prompt_adapter else None

        decoding_config = DecodingConfig(
            guided_decoding_backend=self.guided_decoding_backend)

        detailed_trace_modules = []
        if self.collect_detailed_traces is not None:
            detailed_trace_modules = self.collect_detailed_traces.split(",")
        for m in detailed_trace_modules:
            if m not in ALLOWED_DETAILED_TRACE_MODULES:
                raise ValueError(
                    f"Invalid module {m} in collect_detailed_traces. "
                    f"Valid modules are {ALLOWED_DETAILED_TRACE_MODULES}")
        observability_config = ObservabilityConfig(
            otlp_traces_endpoint=self.otlp_traces_endpoint,
            collect_model_forward_time="model" in detailed_trace_modules
            or "all" in detailed_trace_modules,
            collect_model_execute_time="worker" in detailed_trace_modules
            or "all" in detailed_trace_modules,
        )

        config = VllmConfig(
            model_config=model_config,
            cache_config=cache_config,
            parallel_config=parallel_config,
            scheduler_config=scheduler_config,
            device_config=device_config,
            lora_config=lora_config,
            speculative_config=speculative_config,
            load_config=load_config,
            decoding_config=decoding_config,
            observability_config=observability_config,
            prompt_adapter_config=prompt_adapter_config,
            compilation_config=self.compilation_config,
            kv_transfer_config=self.kv_transfer_config,
        )

        if envs.VLLM_USE_V1:
            self._override_v1_engine_config(config)
        return config
