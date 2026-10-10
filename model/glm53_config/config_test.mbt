///|
fn config_digest() -> @spec.ContentDigest {
  try! @spec.ContentDigest::from_sha256(
    "123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0",
  )
}

///|
fn quoted_schedule(length : Int, value_at : (Int) -> String) -> String {
  [ for index in 0..<length => "\"\{value_at(index)}\"" ].join(",")
}

///|
fn integer_schedule(length : Int, value_at : (Int) -> Int) -> String {
  [ for index in 0..<length => value_at(index).to_string() ].join(",")
}

///|
fn full_config() -> String {
  let mlp = quoted_schedule(78, index => {
    if index < 3 {
      "dense"
    } else {
      "sparse"
    }
  })
  let indexers = quoted_schedule(78, index => {
    if index < 3 || (index - 2) % 4 == 0 {
      "full"
    } else {
      "shared"
    }
  })
  "{\"architectures\":[\"GlmMoeDsaForCausalLM\"],\"model_type\":\"glm_moe_dsa\",\"dtype\":\"bfloat16\",\"vocab_size\":154880,\"hidden_size\":6144,\"intermediate_size\":12288,\"num_hidden_layers\":78,\"num_attention_heads\":64,\"num_key_value_heads\":64,\"max_position_embeddings\":1048576,\"q_lora_rank\":2048,\"kv_lora_rank\":512,\"qk_nope_head_dim\":192,\"qk_rope_head_dim\":64,\"qk_head_dim\":256,\"v_head_dim\":256,\"index_n_heads\":32,\"index_head_dim\":128,\"index_topk\":2048,\"rope_interleave\":true,\"indexer_rope_interleave\":true,\"index_share_for_mtp_iteration\":true,\"n_routed_experts\":256,\"n_shared_experts\":1,\"num_experts_per_tok\":8,\"moe_intermediate_size\":2048,\"routed_scaling_factor\":2.5,\"scoring_func\":\"sigmoid\",\"topk_method\":\"noaux_tc\",\"moe_router_dtype\":\"float32\",\"norm_topk_prob\":true,\"first_k_dense_replace\":3,\"num_nextn_predict_layers\":1,\"rms_norm_eps\":0.00001,\"attention_bias\":false,\"attention_dropout\":0.0,\"tie_word_embeddings\":false,\"use_cache\":true,\"hidden_act\":\"silu\",\"initializer_range\":0.02,\"eos_token_id\":[154820,154827,154829],\"pad_token_id\":154820,\"ep_size\":1,\"head_dim\":192,\"index_skip_topk_offset\":3,\"index_topk_freq\":4,\"index_topk_pattern\":null,\"moe_layer_freq\":1,\"n_group\":1,\"pretraining_tp\":1,\"topk_group\":1,\"transformers_version\":\"5.15.0\",\"mlp_layer_types\":[\{mlp}],\"indexer_types\":[\{indexers}],\"rope_parameters\":{\"rope_theta\":8000000,\"rope_type\":\"default\"},\"quantization_config\":{\"activation_scheme\":\"dynamic\",\"fmt\":\"e4m3\",\"quant_method\":\"fp8\",\"weight_block_size\":[128,128],\"modules_to_not_convert\":[\"lm_head\"]}}"
}

///|
fn flash_config() -> String {
  let mlp = quoted_schedule(45, index => {
    if index < 3 {
      "dense"
    } else {
      "sparse"
    }
  })
  let layers = quoted_schedule(45, index => {
    if index % 4 == 3 {
      "deepseek_sparse_attention"
    } else {
      "linear_attention"
    }
  })
  let indexers = quoted_schedule(45, _ => "full")
  let kda = integer_schedule(34, ordinal => ordinal + ordinal / 3)
  let full = integer_schedule(11, ordinal => 3 + ordinal * 4)
  "{\"architectures\":[\"Glm5NextForConditionalGeneration\"],\"model_type\":\"glm5_next\",\"tie_word_embeddings\":false,\"transformers_version\":\"5.16.0\",\"image_token_id\":154854,\"video_token_id\":154855,\"image_start_token_id\":154830,\"image_end_token_id\":154831,\"video_start_token_id\":154832,\"video_end_token_id\":154833,\"text_config\":{\"model_type\":\"glm5_next_text\",\"dtype\":\"bfloat16\",\"vocab_size\":154880,\"hidden_size\":4096,\"intermediate_size\":12288,\"num_hidden_layers\":45,\"num_attention_heads\":64,\"num_key_value_heads\":64,\"max_position_embeddings\":1048576,\"q_lora_rank\":1536,\"kv_lora_rank\":512,\"qk_nope_head_dim\":256,\"qk_rope_head_dim\":0,\"qk_head_dim\":256,\"v_head_dim\":256,\"index_n_heads\":32,\"index_head_dim\":128,\"index_topk\":2048,\"index_kpool\":4,\"index_kpool_compress\":true,\"index_kpool_always_select_tail\":true,\"indexer_rope_interleave\":true,\"index_share_for_mtp_iteration\":true,\"mla_use_nope\":true,\"mhc\":true,\"hc_mult\":4,\"hc_sinkhorn_iters\":20,\"hc_eps\":0.000001,\"n_routed_experts\":288,\"n_shared_experts\":1,\"num_experts_per_tok\":8,\"moe_intermediate_size\":2048,\"routed_scaling_factor\":2.5,\"scoring_func\":\"sigmoid\",\"topk_method\":\"noaux_tc\",\"moe_router_dtype\":\"float32\",\"norm_topk_prob\":true,\"swiglu_limit\":10.0,\"first_k_dense_replace\":3,\"num_nextn_predict_layers\":1,\"attention_bias\":false,\"attention_dropout\":0.0,\"eos_token_id\":[154820,154827,154829],\"pad_token_id\":154820,\"head_dim\":0,\"hidden_act\":\"silu\",\"initializer_range\":0.02,\"n_group\":1,\"output_router_logits\":false,\"router_aux_loss_coef\":0.001,\"tie_word_embeddings\":false,\"topk_group\":1,\"use_cache\":true,\"mlp_layer_types\":[\{mlp}],\"layer_types\":[\{layers}],\"indexer_types\":[\{indexers}],\"linear_attn_config\":{\"num_heads\":64,\"head_dim\":128,\"short_conv_kernel_size\":4,\"gate_lower_bound\":-5.0,\"kda_layers\":[\{kda}],\"full_attn_layers\":[\{full}]}},\"vision_config\":{\"model_type\":\"glm5_next_vision\",\"depth\":24,\"hidden_size\":1024,\"intermediate_size\":4096,\"num_heads\":16,\"patch_size\":14,\"temporal_patch_size\":2,\"spatial_merge_size\":2,\"out_hidden_size\":4096,\"projection_intermediate_size\":10240,\"image_size\":448,\"in_channels\":3,\"attention_bias\":true,\"attention_dropout\":0.0,\"hidden_act\":\"silu\",\"initializer_range\":0.02,\"rms_norm_eps\":0.00001,\"swiglu_limit\":10.0}}"
}

///|
fn flash_nvfp4_config() -> String {
  let quant =
    #|{"config_groups":{"group_0":{"targets":["Linear"],"weights":{"num_bits":4,"type":"float","group_size":16,"dynamic":false,"symmetric":true},"input_activations":null,"output_activations":null}},"quant_algo":"NVFP4","kv_cache_scheme":null,"producer":{"name":"modelopt","version":"0.45.0"},"quant_method":"modelopt","ignore":["lm_head","model.language_model.embed_tokens","*.self_attn.q_proj","model.visual.*"]}
  flash_config()
  .replace(
    old="\"transformers_version\":\"5.16.0\"",
    new="\"transformers_version\":\"5.16.0\",\"quantization_config\":\{quant}",
  )
  .replace(
    old="\"hc_eps\":0.000001",
    new="\"hc_eps\":0.000001,\"rms_norm_eps\":0.00001",
  )
}

///|
test "Flash accepts installed ModelOpt NVFP4 storage without changing model semantics" {
  let admission = @glm53_config.parse_glm53_config(
    flash_nvfp4_config(),
    config_digest(),
    max_bytes=16384,
    max_depth=16,
  )
  guard admission is Flash(metadata) else {
    fail("Flash profile misclassified")
  }
  assert_eq(metadata.spec().layer_count(), 45)
  assert_eq(metadata.spec().hidden_size(), 4096)
  assert_eq(metadata.spec().rms_norm_epsilon(), 0.00001)
  for
    (old, replacement) in [
      ("\"num_bits\":4", "\"num_bits\":8"),
      ("\"group_size\":16", "\"group_size\":32"),
      ("\"dynamic\":false", "\"dynamic\":true"),
      ("\"quant_algo\":\"NVFP4\"", "\"quant_algo\":\"MXFP4\""),
      ("\"kv_cache_scheme\":null", "\"kv_cache_scheme\":{}"),
      ("\"input_activations\":null", "\"input_activations\":{}"),
      ("\"rms_norm_eps\":0.00001", "\"rms_norm_eps\":0.001"),
      ("\"targets\":[\"Linear\"]", "\"targets\":[\"Conv\"]"),
    ] {
    try
      @glm53_config.parse_glm53_config(
        flash_nvfp4_config().replace(old~, new=replacement),
        config_digest(),
        max_bytes=16384,
        max_depth=16,
      )
    catch {
      InvalidValue(NumericFormat) => ()
      _ => fail("unexpected incompatible numeric metadata failure")
    } noraise {
      _ => fail("incompatible numeric metadata admitted")
    }
  }
}

///|
test "bounded GLM config admission recognizes exact full and Flash profiles" {
  let full = @glm53_config.parse_glm53_config(
    full_config(),
    config_digest(),
    max_bytes=16384,
    max_depth=8,
  )
  let flash = @glm53_config.parse_glm53_config(
    flash_config(),
    config_digest(),
    max_bytes=16384,
    max_depth=8,
  )
  guard full is Full(full_metadata) else { fail("full profile misclassified") }
  guard flash is Flash(flash_metadata) else {
    fail("Flash profile misclassified")
  }
  assert_eq(full_metadata.spec().layer_count(), 78)
  assert_eq(flash_metadata.spec().layer_count(), 45)
  assert_eq(full_metadata.content_digest(), config_digest())
  assert_eq(flash_metadata.content_digest(), config_digest())
  assert_eq(
    @glm53_config.parse_flash_spec_bytes(
      @utf8.encode(flash_config()),
      max_bytes=16384,
      max_depth=8,
    ),
    flash_metadata.spec(),
  )
}

///|
test "spec-only Flash parsing retains profile and byte boundaries" {
  try
    @glm53_config.parse_flash_spec_bytes(
      @utf8.encode(full_config()),
      max_bytes=16384,
      max_depth=8,
    )
  catch {
    InvalidValue(ModelType) => ()
    _ => fail("unexpected profile error")
  } noraise {
    _ => fail("full profile admitted as Flash")
  }
  try
    @glm53_config.parse_flash_spec_bytes(b"{}", max_bytes=1, max_depth=8)
  catch {
    ByteLimitExceeded(limit=1) => ()
    _ => fail("unexpected byte-limit error")
  } noraise {
    _ => fail("byte bound ignored")
  }
}

///|
test "GLM config admission rejects schedule and geometry mutations" {
  for
    document in [
      full_config().replace(
        old="\"hidden_size\":6144",
        new="\"hidden_size\":6145",
      ),
      full_config().replace(
        old="\"shared\",\"shared\",\"shared\"",
        new="\"full\",\"shared\",\"shared\"",
      ),
      flash_config().replace(old="\"hc_mult\":4", new="\"hc_mult\":3"),
      flash_config().replace(
        old="\"gate_lower_bound\":-5.0",
        new="\"gate_lower_bound\":-4.0",
      ),
      flash_config().replace(
        old="\"video_end_token_id\":154833",
        new="\"video_end_token_id\":154832",
      ),
    ] {
    try
      @glm53_config.parse_glm53_config(
        document,
        config_digest(),
        max_bytes=16384,
        max_depth=8,
      )
    catch {
      InvalidValue(_) => ()
      _ => fail("unexpected mutation failure")
    } noraise {
      _ => fail("mutated GLM config admitted")
    }
  }
}

///|
test "GLM config admission rejects duplicate unknown and executable fields" {
  let duplicate = full_config().replace(
    old="\"hidden_size\":6144",
    new="\"hidden_size\":6144,\"hidden_size\":6144",
  )
  try
    @glm53_config.parse_glm53_config(
      duplicate,
      config_digest(),
      max_bytes=16384,
      max_depth=8,
    )
  catch {
    DuplicateField(depth=1, index=_) => ()
    _ => fail("unexpected duplicate failure")
  } noraise {
    _ => fail("duplicate field admitted")
  }
  for insertion in ["\"unexpected\":1,", "\"auto_map\":{},"] {
    let document = flash_config().replace(old="{", new="{\{insertion}")
    try
      @glm53_config.parse_glm53_config(
        document,
        config_digest(),
        max_bytes=16384,
        max_depth=8,
      )
    catch {
      UnknownField(_) | ExecutableMetadata => ()
      _ => fail("unexpected hostile metadata failure")
    } noraise {
      _ => fail("hostile metadata admitted")
    }
  }
}

///|
test "GLM config admission enforces byte and nesting limits" {
  try
    @glm53_config.parse_glm53_config(
      full_config(),
      config_digest(),
      max_bytes=128,
      max_depth=8,
    )
  catch {
    ByteLimitExceeded(limit=128) => ()
    _ => fail("unexpected byte-limit failure")
  } noraise {
    _ => fail("oversized config admitted")
  }
  try
    @glm53_config.parse_glm53_config(
      flash_config(),
      config_digest(),
      max_bytes=16384,
      max_depth=2,
    )
  catch {
    NestingLimitExceeded(limit=2) => ()
    _ => fail("unexpected nesting-limit failure")
  } noraise {
    _ => fail("over-nested config admitted")
  }
}
