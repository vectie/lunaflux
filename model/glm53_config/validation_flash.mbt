///|
fn validate_flash_linear_attention(
  text : Map[String, Json],
  spec : @glm53_spec.Glm53FlashSpec,
) -> Unit raise Glm53ConfigError {
  let linear = require_object(text, "linear_attn_config", LinearAttention)
  validate_known(linear, LinearAttention, known_linear_attention_field)
  check_int(linear, "num_heads", spec.linear_head_count(), LinearAttention)
  check_int(linear, "head_dim", spec.linear_head_dimension(), LinearAttention)
  check_int(
    linear,
    "short_conv_kernel_size",
    spec.linear_conv_kernel_size(),
    LinearAttention,
  )
  check_number(
    linear,
    "gate_lower_bound",
    spec.linear_gate_lower_bound(),
    LinearAttention,
  )
  check_int_schedule(
    linear,
    "kda_layers",
    34,
    ordinal => ordinal + ordinal / 3,
    LayerSchedule,
  )
  check_int_schedule(
    linear,
    "full_attn_layers",
    11,
    ordinal => 3 + ordinal * 4,
    LayerSchedule,
  )
}

///|
fn validate_flash_text(
  text : Map[String, Json],
  spec : @glm53_spec.Glm53FlashSpec,
) -> Unit raise Glm53ConfigError {
  validate_known(text, Profile, known_flash_text_field)
  check_string(text, "model_type", spec.text_model_type(), ModelType)
  check_string(text, "dtype", "bfloat16", NumericFormat)
  check_int(text, "vocab_size", spec.vocabulary_size(), DecoderShape)
  check_int(text, "hidden_size", spec.hidden_size(), DecoderShape)
  check_int(
    text,
    "intermediate_size",
    spec.dense_intermediate_size(),
    DecoderShape,
  )
  check_int(text, "num_hidden_layers", spec.layer_count(), DecoderShape)
  check_int(
    text,
    "num_attention_heads",
    spec.attention_head_count(),
    DecoderShape,
  )
  check_int(text, "num_key_value_heads", spec.kv_head_count(), DecoderShape)
  check_int(
    text,
    "max_position_embeddings",
    spec.context_length(),
    DecoderShape,
  )
  check_int(text, "q_lora_rank", spec.q_lora_rank(), SparseAttention)
  check_int(text, "kv_lora_rank", spec.kv_lora_rank(), SparseAttention)
  check_int(
    text,
    "qk_nope_head_dim",
    spec.qk_nope_head_dimension(),
    SparseAttention,
  )
  check_int(text, "qk_rope_head_dim", 0, SparseAttention)
  check_int(text, "qk_head_dim", spec.qk_nope_head_dimension(), SparseAttention)
  check_int(text, "v_head_dim", spec.value_head_dimension(), SparseAttention)
  check_int(text, "index_n_heads", spec.index_head_count(), SparseAttention)
  check_int(
    text,
    "index_head_dim",
    spec.index_head_dimension(),
    SparseAttention,
  )
  check_int(text, "index_topk", spec.index_topk(), SparseAttention)
  check_int(text, "index_kpool", spec.index_kpool(), SparseAttention)
  check_bool(text, "index_kpool_compress", true, SparseAttention)
  check_bool(text, "index_kpool_always_select_tail", true, SparseAttention)
  check_bool(text, "indexer_rope_interleave", true, SparseAttention)
  check_bool(text, "index_share_for_mtp_iteration", true, SparseAttention)
  check_bool(text, "mla_use_nope", true, SparseAttention)
  check_bool(text, "mhc", true, HyperConnection)
  check_int(text, "hc_mult", spec.hyper_connection_streams(), HyperConnection)
  check_int(
    text,
    "hc_sinkhorn_iters",
    spec.hyper_connection_sinkhorn_iterations(),
    HyperConnection,
  )
  check_number(text, "hc_eps", spec.hyper_connection_epsilon(), HyperConnection)
  check_int(text, "n_routed_experts", spec.routed_expert_count(), ExpertRouting)
  check_int(text, "n_shared_experts", spec.shared_expert_count(), ExpertRouting)
  check_int(
    text,
    "num_experts_per_tok",
    spec.experts_per_token(),
    ExpertRouting,
  )
  check_int(
    text,
    "moe_intermediate_size",
    spec.moe_intermediate_size(),
    ExpertRouting,
  )
  check_number(
    text,
    "routed_scaling_factor",
    spec.routed_scaling_factor(),
    ExpertRouting,
  )
  check_string(text, "scoring_func", "sigmoid", ExpertRouting)
  check_string(text, "topk_method", "noaux_tc", ExpertRouting)
  check_string(text, "moe_router_dtype", "float32", ExpertRouting)
  check_bool(text, "norm_topk_prob", true, ExpertRouting)
  check_number(text, "swiglu_limit", spec.swiglu_limit(), ExpertRouting)
  check_int(text, "first_k_dense_replace", 3, LayerSchedule)
  check_int(text, "num_nextn_predict_layers", 1, MultiTokenPrediction)
  check_bool(text, "attention_bias", false, DecoderShape)
  check_number(text, "attention_dropout", 0.0, DecoderShape)
  check_int_array(text, "eos_token_id", [154820, 154827, 154829], InertMetadata)
  check_int(text, "pad_token_id", 154820, InertMetadata)
  check_int(text, "head_dim", 0, SparseAttention)
  check_string(text, "hidden_act", "silu", DecoderShape)
  check_number(text, "initializer_range", 0.02, InertMetadata)
  check_int(text, "n_group", 1, ExpertRouting)
  check_bool(text, "output_router_logits", false, ExpertRouting)
  check_number(text, "router_aux_loss_coef", 0.001, ExpertRouting)
  check_bool(text, "tie_word_embeddings", false, DecoderShape)
  check_int(text, "topk_group", 1, ExpertRouting)
  check_bool(text, "use_cache", true, InertMetadata)
  check_string_schedule(
    text,
    "mlp_layer_types",
    spec.layer_count(),
    index => if index < 3 { "dense" } else { "sparse" },
    LayerSchedule,
  )
  check_string_schedule(
    text,
    "layer_types",
    spec.layer_count(),
    index => {
      if index % 4 == 3 {
        "deepseek_sparse_attention"
      } else {
        "linear_attention"
      }
    },
    LayerSchedule,
  )
  check_string_schedule(
    text,
    "indexer_types",
    spec.layer_count(),
    _ => "full",
    LayerSchedule,
  )
  validate_flash_linear_attention(text, spec)
}

///|
fn validate_flash_vision(
  vision : Map[String, Json],
  spec : @glm53_spec.Glm53FlashSpec,
) -> Unit raise Glm53ConfigError {
  validate_known(vision, Vision, known_vision_field)
  check_string(vision, "model_type", spec.vision_model_type(), Vision)
  check_int(vision, "depth", spec.vision_layer_count(), Vision)
  check_int(vision, "hidden_size", spec.vision_hidden_size(), Vision)
  check_int(
    vision,
    "intermediate_size",
    spec.vision_intermediate_size(),
    Vision,
  )
  check_int(vision, "num_heads", spec.vision_head_count(), Vision)
  check_int(vision, "patch_size", spec.vision_patch_size(), Vision)
  check_int(
    vision,
    "temporal_patch_size",
    spec.vision_temporal_patch_size(),
    Vision,
  )
  check_int(
    vision,
    "spatial_merge_size",
    spec.vision_spatial_merge_size(),
    Vision,
  )
  check_int(vision, "out_hidden_size", spec.vision_projection_size(), Vision)
  check_int(
    vision,
    "projection_intermediate_size",
    spec.vision_projection_intermediate_size(),
    Vision,
  )
  check_int(vision, "image_size", 448, Vision)
  check_int(vision, "in_channels", 3, Vision)
  check_bool(vision, "attention_bias", true, Vision)
  check_number(vision, "attention_dropout", 0.0, Vision)
  check_string(vision, "hidden_act", "silu", Vision)
  check_number(vision, "initializer_range", 0.02, Vision)
  check_number(vision, "rms_norm_eps", spec.rms_norm_epsilon(), Vision)
  check_number(vision, "swiglu_limit", spec.swiglu_limit(), Vision)
}

///|
fn validate_flash_config(
  root : Map[String, Json],
) -> @glm53_spec.Glm53FlashSpec raise Glm53ConfigError {
  let spec = @glm53_spec.Glm53FlashSpec::official()
  validate_known(root, Profile, known_flash_root_field)
  check_single_architecture(root, spec.architecture())
  check_string(root, "model_type", spec.model_type(), ModelType)
  check_bool(root, "tie_word_embeddings", false, DecoderShape)
  check_string(root, "transformers_version", "5.16.0", InertMetadata)
  validate_flash_text(require_object(root, "text_config", DecoderShape), spec)
  validate_flash_vision(require_object(root, "vision_config", Vision), spec)
  check_int(root, "image_token_id", spec.image_token_id(), ControlTokens)
  check_int(root, "video_token_id", spec.video_token_id(), ControlTokens)
  check_int(
    root,
    "image_start_token_id",
    spec.image_start_token_id(),
    ControlTokens,
  )
  check_int(
    root,
    "image_end_token_id",
    spec.image_end_token_id(),
    ControlTokens,
  )
  check_int(
    root,
    "video_start_token_id",
    spec.video_start_token_id(),
    ControlTokens,
  )
  check_int(
    root,
    "video_end_token_id",
    spec.video_end_token_id(),
    ControlTokens,
  )
  spec
}
