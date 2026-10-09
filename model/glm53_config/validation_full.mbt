///|
fn validate_full_schedule(
  root : Map[String, Json],
  spec : @glm53_spec.Glm53FullSpec,
) -> Unit raise Glm53ConfigError {
  check_string_schedule(
    root,
    "mlp_layer_types",
    spec.layer_count(),
    index => if index < 3 { "dense" } else { "sparse" },
    LayerSchedule,
  )
  check_string_schedule(
    root,
    "indexer_types",
    spec.layer_count(),
    index => {
      if (try! spec.computes_index_topk(index)) {
        "full"
      } else {
        "shared"
      }
    },
    LayerSchedule,
  )
}

///|
fn validate_full_rope(
  root : Map[String, Json],
  spec : @glm53_spec.Glm53FullSpec,
) -> Unit raise Glm53ConfigError {
  let rope = require_object(root, "rope_parameters", SparseAttention)
  validate_known(rope, SparseAttention, known_rope_field)
  check_number(rope, "rope_theta", spec.rope_theta(), SparseAttention)
  check_string(rope, "rope_type", "default", SparseAttention)
}

///|
fn validate_full_quantization(
  root : Map[String, Json],
) -> Unit raise Glm53ConfigError {
  let quant = require_object(root, "quantization_config", NumericFormat)
  validate_known(quant, NumericFormat, known_quantization_field)
  check_string(quant, "activation_scheme", "dynamic", NumericFormat)
  check_string(quant, "fmt", "e4m3", NumericFormat)
  check_string(quant, "quant_method", "fp8", NumericFormat)
  let block = require_int_array(quant, "weight_block_size", NumericFormat)
  if block != [128, 128] {
    raise invalid(NumericFormat)
  }
  let exclusions = require_string_array(
    quant,
    "modules_to_not_convert",
    NumericFormat,
  )
  if exclusions.is_empty() {
    raise invalid(NumericFormat)
  }
}

///|
fn validate_full_config(
  root : Map[String, Json],
) -> @glm53_spec.Glm53FullSpec raise Glm53ConfigError {
  let spec = @glm53_spec.Glm53FullSpec::official()
  validate_known(root, Profile, known_full_field)
  check_single_architecture(root, spec.architecture())
  check_string(root, "model_type", spec.model_type(), ModelType)
  check_string(root, "dtype", "bfloat16", NumericFormat)
  check_int(root, "vocab_size", spec.vocabulary_size(), DecoderShape)
  check_int(root, "hidden_size", spec.hidden_size(), DecoderShape)
  check_int(
    root,
    "intermediate_size",
    spec.dense_intermediate_size(),
    DecoderShape,
  )
  check_int(root, "num_hidden_layers", spec.layer_count(), DecoderShape)
  check_int(
    root,
    "num_attention_heads",
    spec.attention_head_count(),
    DecoderShape,
  )
  check_int(root, "num_key_value_heads", spec.kv_head_count(), DecoderShape)
  check_int(
    root,
    "max_position_embeddings",
    spec.context_length(),
    DecoderShape,
  )
  check_int(root, "q_lora_rank", spec.q_lora_rank(), SparseAttention)
  check_int(root, "kv_lora_rank", spec.kv_lora_rank(), SparseAttention)
  check_int(
    root,
    "qk_nope_head_dim",
    spec.qk_nope_head_dimension(),
    SparseAttention,
  )
  check_int(
    root,
    "qk_rope_head_dim",
    spec.qk_rope_head_dimension(),
    SparseAttention,
  )
  check_int(root, "qk_head_dim", spec.qk_head_dimension(), SparseAttention)
  check_int(root, "v_head_dim", spec.value_head_dimension(), SparseAttention)
  check_int(root, "index_n_heads", spec.index_head_count(), SparseAttention)
  check_int(
    root,
    "index_head_dim",
    spec.index_head_dimension(),
    SparseAttention,
  )
  check_int(root, "index_topk", spec.index_topk(), SparseAttention)
  check_bool(root, "rope_interleave", true, SparseAttention)
  check_bool(root, "indexer_rope_interleave", true, SparseAttention)
  check_bool(root, "index_share_for_mtp_iteration", true, SparseAttention)
  check_int(root, "n_routed_experts", spec.routed_expert_count(), ExpertRouting)
  check_int(root, "n_shared_experts", spec.shared_expert_count(), ExpertRouting)
  check_int(
    root,
    "num_experts_per_tok",
    spec.experts_per_token(),
    ExpertRouting,
  )
  check_int(
    root,
    "moe_intermediate_size",
    spec.moe_intermediate_size(),
    ExpertRouting,
  )
  check_number(
    root,
    "routed_scaling_factor",
    spec.routed_scaling_factor(),
    ExpertRouting,
  )
  check_string(root, "scoring_func", "sigmoid", ExpertRouting)
  check_string(root, "topk_method", "noaux_tc", ExpertRouting)
  check_string(root, "moe_router_dtype", "float32", ExpertRouting)
  check_bool(root, "norm_topk_prob", true, ExpertRouting)
  check_int(root, "first_k_dense_replace", 3, LayerSchedule)
  check_int(
    root,
    "num_nextn_predict_layers",
    spec.mtp_layer_count(),
    MultiTokenPrediction,
  )
  check_number(root, "rms_norm_eps", spec.rms_norm_epsilon(), DecoderShape)
  check_bool(root, "attention_bias", false, DecoderShape)
  check_bool(root, "tie_word_embeddings", false, DecoderShape)
  check_bool(root, "use_cache", true, InertMetadata)
  check_string(root, "hidden_act", "silu", DecoderShape)
  check_number(root, "attention_dropout", 0.0, DecoderShape)
  check_number(root, "initializer_range", 0.02, InertMetadata)
  check_int_array(root, "eos_token_id", [154820, 154827, 154829], InertMetadata)
  check_int(root, "pad_token_id", 154820, InertMetadata)
  check_int(root, "ep_size", 1, ExpertRouting)
  check_int(root, "head_dim", 192, SparseAttention)
  check_int(root, "index_skip_topk_offset", 3, SparseAttention)
  check_int(root, "index_topk_freq", 4, SparseAttention)
  check_null(root, "index_topk_pattern", SparseAttention)
  check_int(root, "moe_layer_freq", 1, LayerSchedule)
  check_int(root, "n_group", 1, ExpertRouting)
  check_int(root, "pretraining_tp", 1, InertMetadata)
  check_int(root, "topk_group", 1, ExpertRouting)
  check_string(root, "transformers_version", "5.15.0", InertMetadata)
  validate_full_schedule(root, spec)
  validate_full_rope(root, spec)
  validate_full_quantization(root)
  spec
}
