///|
fn executable_field(name : StringView) -> Bool {
  name is ("auto_map" | "trust_remote_code" | "custom_pipelines")
}

///|
fn known_full_field(name : StringView) -> Bool {
  name
  is ("architectures"
  | "attention_bias"
  | "attention_dropout"
  | "dtype"
  | "eos_token_id"
  | "ep_size"
  | "first_k_dense_replace"
  | "head_dim"
  | "hidden_act"
  | "hidden_size"
  | "index_head_dim"
  | "index_n_heads"
  | "index_share_for_mtp_iteration"
  | "index_skip_topk_offset"
  | "index_topk"
  | "index_topk_freq"
  | "index_topk_pattern"
  | "indexer_rope_interleave"
  | "indexer_types"
  | "initializer_range"
  | "intermediate_size"
  | "kv_lora_rank"
  | "max_position_embeddings"
  | "mlp_layer_types"
  | "model_type"
  | "moe_intermediate_size"
  | "moe_layer_freq"
  | "moe_router_dtype"
  | "n_group"
  | "n_routed_experts"
  | "n_shared_experts"
  | "norm_topk_prob"
  | "num_attention_heads"
  | "num_experts_per_tok"
  | "num_hidden_layers"
  | "num_key_value_heads"
  | "num_nextn_predict_layers"
  | "pad_token_id"
  | "pretraining_tp"
  | "q_lora_rank"
  | "qk_head_dim"
  | "qk_nope_head_dim"
  | "qk_rope_head_dim"
  | "quantization_config"
  | "rms_norm_eps"
  | "rope_interleave"
  | "rope_parameters"
  | "routed_scaling_factor"
  | "scoring_func"
  | "tie_word_embeddings"
  | "topk_group"
  | "topk_method"
  | "transformers_version"
  | "use_cache"
  | "v_head_dim"
  | "vocab_size")
}

///|
fn known_flash_root_field(name : StringView) -> Bool {
  name
  is ("architectures"
  | "text_config"
  | "vision_config"
  | "image_token_id"
  | "video_token_id"
  | "image_start_token_id"
  | "image_end_token_id"
  | "video_start_token_id"
  | "video_end_token_id"
  | "tie_word_embeddings"
  | "model_type"
  | "transformers_version")
}

///|
fn known_flash_text_field(name : StringView) -> Bool {
  known_full_field(name) ||
  name
  is ("hc_eps"
  | "hc_mult"
  | "hc_sinkhorn_iters"
  | "index_kpool"
  | "index_kpool_always_select_tail"
  | "index_kpool_compress"
  | "layer_types"
  | "linear_attn_config"
  | "mhc"
  | "mla_use_nope"
  | "output_router_logits"
  | "router_aux_loss_coef"
  | "swiglu_limit")
}

///|
fn known_vision_field(name : StringView) -> Bool {
  name
  is ("attention_bias"
  | "attention_dropout"
  | "depth"
  | "hidden_act"
  | "hidden_size"
  | "image_size"
  | "in_channels"
  | "initializer_range"
  | "intermediate_size"
  | "model_type"
  | "num_heads"
  | "out_hidden_size"
  | "patch_size"
  | "projection_intermediate_size"
  | "rms_norm_eps"
  | "spatial_merge_size"
  | "swiglu_limit"
  | "temporal_patch_size")
}

///|
fn known_rope_field(name : StringView) -> Bool {
  name is ("rope_theta" | "rope_type")
}

///|
fn known_quantization_field(name : StringView) -> Bool {
  name
  is ("activation_scheme"
  | "fmt"
  | "quant_method"
  | "weight_block_size"
  | "modules_to_not_convert")
}

///|
fn known_linear_attention_field(name : StringView) -> Bool {
  name
  is ("num_heads"
  | "gate_lower_bound"
  | "head_dim"
  | "short_conv_kernel_size"
  | "kda_layers"
  | "full_attn_layers")
}

///|
fn validate_known(
  object : Map[String, Json],
  component : Glm53ConfigField,
  predicate : (StringView) -> Bool,
) -> Unit raise Glm53ConfigError {
  let mut index = 0
  for name, _ in object {
    if executable_field(name) {
      raise ExecutableMetadata
    }
    if !predicate(name) {
      raise UnknownField(component~, index~)
    }
    index += 1
  }
}
