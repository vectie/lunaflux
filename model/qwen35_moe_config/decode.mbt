///|
fn validate_text(text : Map[String, Json]) -> Unit raise Qwen35ConfigError {
  validate_known_fields(text, Text)
  expect_bool(text, Text, "attention_bias", AttentionBias, false)
  expect_number(text, Text, "attention_dropout", AttentionDropout, 0.0)
  expect_bool(text, Text, "attn_output_gate", AttentionOutputGate, true)
  expect_int(text, Text, "bos_token_id", TokenId, 248044)
  expect_string(text, Text, "dtype", WeightDType, "bfloat16")
  expect_int(text, Text, "eos_token_id", TokenId, 248044)
  expect_int(text, Text, "full_attention_interval", FullAttentionInterval, 4)
  expect_int(text, Text, "head_dim", HeadDimension, 256)
  expect_string(text, Text, "hidden_act", HiddenActivation, "silu")
  expect_int(text, Text, "hidden_size", HiddenSize, 2048)
  expect_number(text, Text, "initializer_range", InitializerRange, 0.02)
  validate_layer_types(text)
  expect_int(text, Text, "linear_conv_kernel_dim", LinearConvWidth, 4)
  expect_int(text, Text, "linear_key_head_dim", LinearKeyHeadDimension, 128)
  expect_int(text, Text, "linear_num_key_heads", LinearKeyHeadCount, 16)
  expect_int(text, Text, "linear_num_value_heads", LinearValueHeadCount, 32)
  expect_int(text, Text, "linear_value_head_dim", LinearValueHeadDimension, 128)
  expect_string(text, Text, "mamba_ssm_dtype", RecurrentStateDType, "float32")
  expect_int(text, Text, "max_position_embeddings", ContextLength, 262144)
  expect_string(text, Text, "model_type", TextModelType, "qwen3_5_moe_text")
  expect_int(text, Text, "moe_intermediate_size", ExpertIntermediateSize, 512)
  expect_int(text, Text, "mtp_num_hidden_layers", MtpLayerCount, 1)
  expect_bool(
    text,
    Text,
    "mtp_use_dedicated_embeddings",
    MtpDedicatedEmbeddings,
    false,
  )
  expect_int(text, Text, "num_attention_heads", AttentionHeadCount, 16)
  expect_int(text, Text, "num_experts", RoutedExpertCount, 256)
  expect_int(text, Text, "num_experts_per_tok", ExpertsPerToken, 8)
  expect_int(text, Text, "num_hidden_layers", LayerCount, 40)
  expect_int(text, Text, "num_key_value_heads", KvHeadCount, 2)
  expect_bool(text, Text, "output_router_logits", OutputRouterLogits, false)
  require_null(text, Text, "pad_token_id", PadTokenId)
  expect_number(text, Text, "partial_rotary_factor", PartialRotaryFactor, 0.25)
  expect_number(text, Text, "rms_norm_eps", RmsNormEpsilon, 0.000001)
  validate_rope(require_object(text, Text, "rope_parameters", RopeParameters))
  expect_number(text, Text, "router_aux_loss_coef", RouterAuxLoss, 0.001)
  expect_int(
    text,
    Text,
    "shared_expert_intermediate_size",
    SharedExpertIntermediateSize,
    512,
  )
  expect_bool(text, Text, "tie_word_embeddings", TieWordEmbeddings, false)
  expect_bool(text, Text, "use_cache", UseCache, true)
  expect_int(text, Text, "vocab_size", VocabularySize, 248320)
}

///|
fn decode_root(
  root : Map[String, Json],
  content_digest : @spec.ContentDigest,
) -> @qwen35_moe_spec.Qwen35MoeTextMetadata raise Qwen35ConfigError {
  validate_known_fields(root, Root)
  expect_single_architecture(root)
  expect_string(root, Root, "model_type", RootModelType, "qwen3_5_moe")
  expect_bool(root, Root, "tie_word_embeddings", TieWordEmbeddings, false)
  expect_int(root, Root, "image_token_id", TokenId, 248056)
  expect_int(root, Root, "video_token_id", TokenId, 248057)
  expect_int(root, Root, "vision_start_token_id", TokenId, 248053)
  expect_int(root, Root, "vision_end_token_id", TokenId, 248054)
  ignore(
    require_string(root, Root, "transformers_version", TransformersVersion),
  )
  // The exact artifact contains vision metadata, but this route intentionally
  // grants no vision execution authority.
  ignore(require_object(root, Root, "vision_config", VisionConfig))
  validate_quantization(
    require_object(root, Root, "quantization_config", QuantizationConfig),
  )
  validate_text(require_object(root, Root, "text_config", TextConfig))
  @qwen35_moe_spec.Qwen35MoeTextMetadata::from_verified_config(content_digest)
}
