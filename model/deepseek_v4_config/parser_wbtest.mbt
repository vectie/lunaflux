///|
fn test_content() -> @spec.ContentDigest {
  try! @spec.ContentDigest::from_sha256(
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  )
}

///|
fn flash_config() -> String {
  let document =
    #|{"architectures":["DeepseekV4ForCausalLM"],"attention_bias":false,"attention_dropout":0.0,"bos_token_id":0,"eos_token_id":1,"expert_dtype":"fp4","hc_eps":0.000001,"hc_mult":4,"hc_sinkhorn_iters":20,"head_dim":512,"hidden_act":"silu","hidden_size":4096,"index_head_dim":128,"index_n_heads":64,"index_topk":512,"initializer_range":0.02,"max_position_embeddings":1048576,"model_type":"deepseek_v4","moe_intermediate_size":2048,"n_routed_experts":256,"n_shared_experts":1,"norm_topk_prob":true,"num_attention_heads":64,"num_experts_per_tok":6,"num_hidden_layers":43,"num_hash_layers":3,"num_key_value_heads":1,"num_nextn_predict_layers":1,"o_groups":8,"o_lora_rank":1024,"q_lora_rank":1024,"qk_rope_head_dim":64,"quantization_config":{"activation_scheme":"dynamic","fmt":"e4m3","quant_method":"fp8","scale_fmt":"ue8m0","weight_block_size":[128,128]},"rms_norm_eps":0.000001,"rope_scaling":{"beta_fast":32,"beta_slow":1,"factor":16,"original_max_position_embeddings":65536,"type":"yarn"},"rope_theta":10000,"routed_scaling_factor":1.5,"scoring_func":"sqrtsoftplus","sliding_window":128,"swiglu_limit":10.0,"tie_word_embeddings":false,"topk_method":"noaux_tc","torch_dtype":"bfloat16","transformers_version":"4.57.1","use_cache":true,"vocab_size":129280,"compress_rope_theta":160000,"compress_ratios":[0,0,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,0]}
  document
}

///|
fn pro_config() -> String {
  let document =
    #|{"architectures":["DeepseekV4ForCausalLM"],"attention_bias":false,"attention_dropout":0.0,"bos_token_id":0,"eos_token_id":1,"expert_dtype":"fp4","hc_eps":0.000001,"hc_mult":4,"hc_sinkhorn_iters":20,"head_dim":512,"hidden_act":"silu","hidden_size":7168,"index_head_dim":128,"index_n_heads":64,"index_topk":1024,"initializer_range":0.02,"max_position_embeddings":1048576,"model_type":"deepseek_v4","moe_intermediate_size":3072,"n_routed_experts":384,"n_shared_experts":1,"norm_topk_prob":true,"num_attention_heads":128,"num_experts_per_tok":6,"num_hidden_layers":61,"num_hash_layers":3,"num_key_value_heads":1,"num_nextn_predict_layers":1,"o_groups":16,"o_lora_rank":1024,"q_lora_rank":1536,"qk_rope_head_dim":64,"quantization_config":{"activation_scheme":"dynamic","fmt":"e4m3","quant_method":"fp8","scale_fmt":"ue8m0","weight_block_size":[128,128]},"rms_norm_eps":0.000001,"rope_scaling":{"beta_fast":32,"beta_slow":1,"factor":16,"original_max_position_embeddings":65536,"type":"yarn"},"rope_theta":10000,"routed_scaling_factor":2.5,"scoring_func":"sqrtsoftplus","sliding_window":128,"swiglu_limit":10.0,"tie_word_embeddings":false,"topk_method":"noaux_tc","torch_dtype":"bfloat16","transformers_version":"4.57.1","use_cache":true,"vocab_size":129280,"compress_rope_theta":160000,"compress_ratios":[128,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,0]}
  document
}

///|
fn flash_0731_config() -> String {
  let document =
    #|{"architectures":["DeepseekV4ForCausalLM"],"attention_bias":false,"attention_dropout":0.0,"bos_token_id":0,"eos_token_id":1,"expert_dtype":"fp4","hc_eps":0.000001,"hc_mult":4,"hc_sinkhorn_iters":20,"head_dim":512,"hidden_act":"silu","hidden_size":4096,"index_head_dim":128,"index_n_heads":64,"index_topk":512,"initializer_range":0.02,"max_position_embeddings":1048576,"model_type":"deepseek_v4","moe_intermediate_size":2048,"n_routed_experts":256,"n_shared_experts":1,"norm_topk_prob":true,"num_attention_heads":64,"num_experts_per_tok":6,"num_hidden_layers":43,"num_hash_layers":3,"num_key_value_heads":1,"num_nextn_predict_layers":1,"o_groups":8,"o_lora_rank":1024,"q_lora_rank":1024,"qk_rope_head_dim":64,"quantization_config":{"activation_scheme":"dynamic","fmt":"e4m3","quant_method":"fp8","scale_fmt":"ue8m0","weight_block_size":[128,128]},"rms_norm_eps":0.000001,"rope_scaling":{"beta_fast":32,"beta_slow":1,"factor":16,"original_max_position_embeddings":65536,"type":"yarn"},"rope_theta":10000,"routed_scaling_factor":1.5,"scoring_func":"sqrtsoftplus","sliding_window":128,"swiglu_limit":10.0,"tie_word_embeddings":false,"topk_method":"noaux_tc","torch_dtype":"bfloat16","transformers_version":"4.57.1","use_cache":true,"vocab_size":129280,"compress_rope_theta":160000,"compress_ratios":[0,0,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,0,0,0],"dspark_block_size":5,"dspark_noise_token_id":128799,"dspark_target_layer_ids":[40,41,42],"dspark_markov_rank":256}
  document
}

///|
fn parse_test(
  document : StringView,
) -> @deepseek_v4_spec.DeepSeekV4ModelMetadata raise DeepSeekV4ConfigError {
  parse_string(document, test_content(), max_bytes=8192, max_depth=8)
}

///|
test "all five exact official DeepSeek V4 profiles are closed" {
  let cases : Array[(String, @deepseek_v4_spec.DeepSeekV4Profile)] = [
    (flash_config(), Flash),
    (
      flash_config().replace(
        old="\"expert_dtype\":\"fp4\"",
        new="\"expert_dtype\":\"fp8\"",
      ),
      FlashBase,
    ),
    (pro_config(), Pro),
    (
      pro_config().replace(
        old="\"expert_dtype\":\"fp4\"",
        new="\"expert_dtype\":\"fp8\"",
      ),
      ProBase,
    ),
    (flash_0731_config(), Flash0731),
  ]
  for pair in cases {
    let (document, profile) = pair
    let metadata = parse_test(document)
    assert_eq(metadata.spec().profile(), profile)
    assert_eq(metadata.content_digest(), test_content())
    let bytes_metadata = parse_bytes(
      @utf8.encode(document),
      test_content(),
      max_bytes=8192,
      max_depth=8,
    )
    assert_eq(bytes_metadata, metadata)
    assert_eq(
      parse_spec_bytes(@utf8.encode(document), max_bytes=8192, max_depth=8),
      metadata.spec(),
    )
  }
}

///|
test "spec-only parsing retains byte and schema boundaries" {
  try parse_spec_bytes(b"{}", max_bytes=1, max_depth=8) catch {
    ByteLimitExceeded(1) => ()
    _ => fail("unexpected byte-limit error")
  } noraise {
    _ => fail("byte bound ignored")
  }
  try parse_spec_bytes(b"{}", max_bytes=8192, max_depth=8) catch {
    MissingField(_) => ()
    _ => fail("unexpected schema error")
  } noraise {
    _ => fail("missing execution semantics admitted")
  }
}

///|
test "hostile semantic mutations and schema widening fail closed" {
  let source = flash_config()
  let mutations = [
    source.replace(old="\"index_topk\":512", new="\"index_topk\":511"),
    source.replace(old="\"sqrtsoftplus\"", new="\"sigmoid\""),
    source.replace(old="[128,128]", new="[128,64]"),
    source.replace(old="\"beta_fast\":32", new="\"beta_fast\":31"),
    source.replace(old="\"use_cache\":true", new="\"use_cache\":false"),
  ]
  for mutation in mutations {
    try parse_test(mutation) catch {
      InvalidField(_) | UnsupportedProfile => ()
      _ => fail("unexpected hostile-mutation error")
    } noraise {
      _ => fail("mutated official semantics were admitted")
    }
  }
  try
    parse_test(
      source.replace(
        old="\"use_cache\":true",
        new="\"use_cache\":true,\"future_mode\":1",
      ),
    )
  catch {
    UnknownField(_) => ()
    _ => fail("unexpected unknown-field error")
  } noraise {
    _ => fail("unknown config field was admitted")
  }
  try
    parse_test(
      source.replace(
        old="\"use_cache\":true",
        new="\"use_cache\":true,\"auto_map\":{}",
      ),
    )
  catch {
    ExecutableMetadata(0) => ()
    _ => fail("unexpected executable-metadata error")
  } noraise {
    _ => fail("remote-code metadata was admitted")
  }
}

///|
test "duplicate keys and resource bounds reject before admission" {
  let source = flash_config()
  try
    parse_test(
      source.replace(
        old="\"model_type\":\"deepseek_v4\"",
        new="\"model_type\":\"deepseek_v4\",\"model_type\":\"deepseek_v4\"",
      ),
    )
  catch {
    DuplicateField(_, _) => ()
    _ => fail("unexpected duplicate-field error")
  } noraise {
    _ => fail("duplicate semantic key was admitted")
  }
  try parse_string(source, test_content(), max_bytes=64, max_depth=8) catch {
    ByteLimitExceeded(64) => ()
    _ => fail("unexpected byte-limit error")
  } noraise {
    _ => fail("oversized config was admitted")
  }
  try parse_string(source, test_content(), max_bytes=8192, max_depth=1) catch {
    NestingLimitExceeded(1) => ()
    _ => fail("unexpected depth-limit error")
  } noraise {
    _ => fail("over-deep config was admitted")
  }
}
