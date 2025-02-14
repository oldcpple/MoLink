from vllm.model_executor.models.registry import _ModelRegistry, _LazyRegisteredModel

_TEXT_GENERATION_MODELS = {
    # [Decoder-only]
    "AquilaModel": ("llama", "LlamaForCausalLM"),
    "AquilaForCausalLM": ("llama", "LlamaForCausalLM"),  # AquilaChat2
    "ArcticForCausalLM": ("arctic", "ArcticForCausalLM"),
    # baichuan-7b, upper case 'C' in the class name
    "BaiChuanForCausalLM": ("baichuan", "BaiChuanForCausalLM"),
    # baichuan-13b, lower case 'c' in the class name
    "BaichuanForCausalLM": ("baichuan", "BaichuanForCausalLM"),
    "BloomForCausalLM": ("bloom", "BloomForCausalLM"),
    # ChatGLMModel supports multimodal
    "CohereForCausalLM": ("commandr", "CohereForCausalLM"),
    "Cohere2ForCausalLM": ("commandr", "CohereForCausalLM"),
    "DbrxForCausalLM": ("dbrx", "DbrxForCausalLM"),
    "DeciLMForCausalLM": ("decilm", "DeciLMForCausalLM"),
    "DeepseekForCausalLM": ("deepseek", "DeepseekForCausalLM"),
    "DeepseekV2ForCausalLM": ("deepseek_v2", "DeepseekV2ForCausalLM"),
    "DeepseekV3ForCausalLM": ("deepseek_v3", "DeepseekV3ForCausalLM"),
    "ExaoneForCausalLM": ("exaone", "ExaoneForCausalLM"),
    "FalconForCausalLM": ("falcon", "FalconForCausalLM"),
    "GemmaForCausalLM": ("gemma", "GemmaForCausalLM"),
    "Gemma2ForCausalLM": ("gemma2", "Gemma2ForCausalLM"),
    "GlmForCausalLM": ("glm", "GlmForCausalLM"),
    "GPT2LMHeadModel": ("gpt2", "GPT2LMHeadModel"),
    "GPTBigCodeForCausalLM": ("gpt_bigcode", "GPTBigCodeForCausalLM"),
    "GPTJForCausalLM": ("gpt_j", "GPTJForCausalLM"),
    "GPTNeoXForCausalLM": ("gpt_neox", "GPTNeoXForCausalLM"),
    "GraniteForCausalLM": ("granite", "GraniteForCausalLM"),
    "GraniteMoeForCausalLM": ("granitemoe", "GraniteMoeForCausalLM"),
    "GritLM": ("gritlm", "GritLM"),
    "InternLMForCausalLM": ("llama", "LlamaForCausalLM"),
    "InternLM2ForCausalLM": ("internlm2", "InternLM2ForCausalLM"),
    "InternLM2VEForCausalLM": ("internlm2_ve", "InternLM2VEForCausalLM"),
    "InternLM3ForCausalLM": ("llama", "LlamaForCausalLM"),
    "JAISLMHeadModel": ("jais", "JAISLMHeadModel"),
    "JambaForCausalLM": ("jamba", "JambaForCausalLM"),
    "LlamaForCausalLM": ("llama", "LlamaForCausalLM"),
    # For decapoda-research/llama-*
    "LLaMAForCausalLM": ("llama", "LlamaForCausalLM"),
    "MambaForCausalLM": ("mamba", "MambaForCausalLM"),
    "FalconMambaForCausalLM": ("mamba", "MambaForCausalLM"),
    "MiniCPMForCausalLM": ("minicpm", "MiniCPMForCausalLM"),
    "MiniCPM3ForCausalLM": ("minicpm3", "MiniCPM3ForCausalLM"),
    "MistralForCausalLM": ("llama", "LlamaForCausalLM"),
    "MixtralForCausalLM": ("mixtral", "MixtralForCausalLM"),
    "QuantMixtralForCausalLM": ("mixtral_quant", "MixtralForCausalLM"),
    # transformers's mpt class has lower case
    "MptForCausalLM": ("mpt", "MPTForCausalLM"),
    "MPTForCausalLM": ("mpt", "MPTForCausalLM"),
    "NemotronForCausalLM": ("nemotron", "NemotronForCausalLM"),
    "OlmoForCausalLM": ("olmo", "OlmoForCausalLM"),
    "Olmo2ForCausalLM": ("olmo2", "Olmo2ForCausalLM"),
    "OlmoeForCausalLM": ("olmoe", "OlmoeForCausalLM"),
    "OPTForCausalLM": ("opt", "OPTForCausalLM"),
    "OrionForCausalLM": ("orion", "OrionForCausalLM"),
    "PersimmonForCausalLM": ("persimmon", "PersimmonForCausalLM"),
    "PhiForCausalLM": ("phi", "PhiForCausalLM"),
    "Phi3ForCausalLM": ("phi3", "Phi3ForCausalLM"),
    "Phi3SmallForCausalLM": ("phi3_small", "Phi3SmallForCausalLM"),
    "PhiMoEForCausalLM": ("phimoe", "PhiMoEForCausalLM"),
    # QWenLMHeadModel supports multimodal
    "Qwen2ForCausalLM": ("qwen2", "Qwen2ForCausalLM"),
    "Qwen2MoeForCausalLM": ("qwen2_moe", "Qwen2MoeForCausalLM"),
    "RWForCausalLM": ("falcon", "FalconForCausalLM"),
    "StableLMEpochForCausalLM": ("stablelm", "StablelmForCausalLM"),
    "StableLmForCausalLM": ("stablelm", "StablelmForCausalLM"),
    "Starcoder2ForCausalLM": ("starcoder2", "Starcoder2ForCausalLM"),
    "SolarForCausalLM": ("solar", "SolarForCausalLM"),
    "TeleChat2ForCausalLM": ("telechat2", "TeleChat2ForCausalLM"),
    "XverseForCausalLM": ("llama", "LlamaForCausalLM"),
    # [Encoder-decoder]
    "BartModel": ("bart", "BartForConditionalGeneration"),
    "BartForConditionalGeneration": ("bart", "BartForConditionalGeneration"),
    "Florence2ForConditionalGeneration": ("florence2", "Florence2ForConditionalGeneration"),  # noqa: E501
}

_EMBEDDING_MODELS = {
    # [Text-only]
    "BertModel": ("bert", "BertEmbeddingModel"),
    "RobertaModel": ("roberta", "RobertaEmbeddingModel"),
    "RobertaForMaskedLM": ("roberta", "RobertaEmbeddingModel"),
    "XLMRobertaModel": ("roberta", "RobertaEmbeddingModel"),
    "DeciLMForCausalLM": ("decilm", "DeciLMForCausalLM"),
    "Gemma2Model": ("gemma2", "Gemma2ForCausalLM"),
    "GlmForCausalLM": ("glm", "GlmForCausalLM"),
    "GritLM": ("gritlm", "GritLM"),
    "InternLM2ForRewardModel": ("internlm2", "InternLM2ForRewardModel"),
    "JambaForSequenceClassification": ("jamba", "JambaForSequenceClassification"),  # noqa: E501
    "LlamaModel": ("llama", "LlamaForCausalLM"),
    **{
        # Multiple models share the same architecture, so we include them all
        k: (mod, arch) for k, (mod, arch) in _TEXT_GENERATION_MODELS.items()
        if arch == "LlamaForCausalLM"
    },
    "MistralModel": ("llama", "LlamaForCausalLM"),
    "Phi3ForCausalLM": ("phi3", "Phi3ForCausalLM"),
    "Qwen2Model": ("qwen2", "Qwen2EmbeddingModel"),
    "Qwen2ForCausalLM": ("qwen2", "Qwen2ForCausalLM"),
    "Qwen2ForRewardModel": ("qwen2_rm", "Qwen2ForRewardModel"),
    "TeleChat2ForCausalLM": ("telechat2", "TeleChat2ForCausalLM"),
    # [Multimodal]
    "LlavaNextForConditionalGeneration": ("llava_next", "LlavaNextForConditionalGeneration"),  # noqa: E501
    "Phi3VForCausalLM": ("phi3v", "Phi3VForCausalLM"),
    "Qwen2VLForConditionalGeneration": ("qwen2_vl", "Qwen2VLForConditionalGeneration"),  # noqa: E501
    # [Auto-converted (see adapters.py)]
    "Qwen2ForSequenceClassification": ("qwen2", "Qwen2ForCausalLM"),
}

_CROSS_ENCODER_MODELS = {
    "BertForSequenceClassification": ("bert", "BertForSequenceClassification"),
    "RobertaForSequenceClassification": ("roberta",
                                         "RobertaForSequenceClassification"),
    "XLMRobertaForSequenceClassification": ("roberta",
                                            "RobertaForSequenceClassification"),
}

_MULTIMODAL_MODELS = {
    # [Decoder-only]
    "AriaForConditionalGeneration": ("aria", "AriaForConditionalGeneration"),
    "Blip2ForConditionalGeneration": ("blip2", "Blip2ForConditionalGeneration"),
    "ChameleonForConditionalGeneration": ("chameleon", "ChameleonForConditionalGeneration"),  # noqa: E501
    "ChatGLMModel": ("chatglm", "ChatGLMForCausalLM"),
    "ChatGLMForConditionalGeneration": ("chatglm", "ChatGLMForCausalLM"),
    "DeepseekVLV2ForCausalLM": ("deepseek_vl2", "DeepseekVLV2ForCausalLM"),
    "FuyuForCausalLM": ("fuyu", "FuyuForCausalLM"),
    "H2OVLChatModel": ("h2ovl", "H2OVLChatModel"),
    "InternVLChatModel": ("internvl", "InternVLChatModel"),
    "Idefics3ForConditionalGeneration":("idefics3","Idefics3ForConditionalGeneration"),
    "LlavaForConditionalGeneration": ("llava", "LlavaForConditionalGeneration"),
    "LlavaNextForConditionalGeneration": ("llava_next", "LlavaNextForConditionalGeneration"),  # noqa: E501
    "LlavaNextVideoForConditionalGeneration": ("llava_next_video", "LlavaNextVideoForConditionalGeneration"),  # noqa: E501
    "LlavaOnevisionForConditionalGeneration": ("llava_onevision", "LlavaOnevisionForConditionalGeneration"),  # noqa: E501
    "MantisForConditionalGeneration": ("llava", "MantisForConditionalGeneration"),  # noqa: E501
    "MiniCPMV": ("minicpmv", "MiniCPMV"),
    "MolmoForCausalLM": ("molmo", "MolmoForCausalLM"),
    "NVLM_D": ("nvlm_d", "NVLM_D_Model"),
    "PaliGemmaForConditionalGeneration": ("paligemma", "PaliGemmaForConditionalGeneration"),  # noqa: E501
    "Phi3VForCausalLM": ("phi3v", "Phi3VForCausalLM"),
    "PixtralForConditionalGeneration": ("pixtral", "PixtralForConditionalGeneration"),  # noqa: E501
    "QWenLMHeadModel": ("qwen", "QWenLMHeadModel"),
    "Qwen2VLForConditionalGeneration": ("qwen2_vl", "Qwen2VLForConditionalGeneration"),  # noqa: E501
    "Qwen2AudioForConditionalGeneration": ("qwen2_audio", "Qwen2AudioForConditionalGeneration"),  # noqa: E501
    "UltravoxModel": ("ultravox", "UltravoxModel"),
    # [Encoder-decoder]
    "MllamaForConditionalGeneration": ("mllama", "MllamaForConditionalGeneration"),  # noqa: E501
    "WhisperForConditionalGeneration": ("whisper", "WhisperForConditionalGeneration"),  # noqa: E501
}

_SPECULATIVE_DECODING_MODELS = {
    "EAGLEModel": ("eagle", "EAGLE"),
    "MedusaModel": ("medusa", "Medusa"),
    "MLPSpeculatorPreTrainedModel": ("mlp_speculator", "MLPSpeculator"),
}
# yapf: enable

_MOLINK_MODELS = {
    **_TEXT_GENERATION_MODELS,
    **_EMBEDDING_MODELS,
    **_CROSS_ENCODER_MODELS,
    **_MULTIMODAL_MODELS,
    **_SPECULATIVE_DECODING_MODELS,
}

ModelRegistry = _ModelRegistry({
    model_arch: _LazyRegisteredModel(
        module_name=f"molink.model_executor.models.{mod_relname}",
        class_name=cls_name,
    )
    for model_arch, (mod_relname, cls_name) in _MOLINK_MODELS.items()
})