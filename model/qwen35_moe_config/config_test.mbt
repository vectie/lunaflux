///|
fn config_digest() -> @spec.ContentDigest {
  try! @spec.ContentDigest::from_sha256(
    "570ef7ea45a7e1d3de2b1d3c70c4ac3562d0e768acdc195778cb4f4d95025845",
  )
}

///|
fn layer_types_json() -> String {
  (
    #|["linear_attention","linear_attention","linear_attention","full_attention",
    #|"linear_attention","linear_attention","linear_attention","full_attention",
    #|"linear_attention","linear_attention","linear_attention","full_attention",
    #|"linear_attention","linear_attention","linear_attention","full_attention",
    #|"linear_attention","linear_attention","linear_attention","full_attention",
    #|"linear_attention","linear_attention","linear_attention","full_attention",
    #|"linear_attention","linear_attention","linear_attention","full_attention",
    #|"linear_attention","linear_attention","linear_attention","full_attention",
    #|"linear_attention","linear_attention","linear_attention","full_attention",
    #|"linear_attention","linear_attention","linear_attention","full_attention"]
  )
}

///|
fn selected_qwen35_config() -> String {
  (
    #|{
    #|  "architectures":["Qwen3_5MoeForConditionalGeneration"],
    #|  "image_token_id":248056,
    #|  "model_type":"qwen3_5_moe",
    #|  "quantization_config":{
    #|    "activation_scheme":"dynamic",
    #|    "fmt":"e4m3",
    #|    "modules_to_not_convert":["model.embed_tokens","lm_head"],
    #|    "quant_method":"fp8",
    #|    "weight_block_size":[128,128]
    #|  },
    #|  "text_config":{
    #|    "attention_bias":false,
    #|    "attention_dropout":0.0,
    #|    "attn_output_gate":true,
    #|    "bos_token_id":248044,
    #|    "dtype":"bfloat16",
    #|    "eos_token_id":248044,
    #|    "full_attention_interval":4,
    #|    "head_dim":256,
    #|    "hidden_act":"silu",
    #|    "hidden_size":2048,
    #|    "initializer_range":0.02,
    #|    "layer_types":
  ) +
  layer_types_json() +
  (
    #|,
    #|    "linear_conv_kernel_dim":4,
    #|    "linear_key_head_dim":128,
    #|    "linear_num_key_heads":16,
    #|    "linear_num_value_heads":32,
    #|    "linear_value_head_dim":128,
    #|    "mamba_ssm_dtype":"float32",
    #|    "max_position_embeddings":262144,
    #|    "model_type":"qwen3_5_moe_text",
    #|    "moe_intermediate_size":512,
    #|    "mtp_num_hidden_layers":1,
    #|    "mtp_use_dedicated_embeddings":false,
    #|    "num_attention_heads":16,
    #|    "num_experts":256,
    #|    "num_experts_per_tok":8,
    #|    "num_hidden_layers":40,
    #|    "num_key_value_heads":2,
    #|    "output_router_logits":false,
    #|    "pad_token_id":null,
    #|    "partial_rotary_factor":0.25,
    #|    "rms_norm_eps":0.000001,
    #|    "rope_parameters":{
    #|      "mrope_interleaved":true,
    #|      "mrope_section":[11,11,10],
    #|      "partial_rotary_factor":0.25,
    #|      "rope_theta":10000000,
    #|      "rope_type":"default"
    #|    },
    #|    "router_aux_loss_coef":0.001,
    #|    "shared_expert_intermediate_size":512,
    #|    "tie_word_embeddings":false,
    #|    "use_cache":true,
    #|    "vocab_size":248320
    #|  },
    #|  "tie_word_embeddings":false,
    #|  "transformers_version":"4.57.1",
    #|  "video_token_id":248057,
    #|  "vision_config":{},
    #|  "vision_end_token_id":248054,
    #|  "vision_start_token_id":248053
    #|}
  )
}

///|
fn parse_selected(
  document : StringView,
) -> @qwen35_moe_spec.Qwen35MoeTextMetadata raise Qwen35ConfigError {
  @qwen35_moe_config.parse_string(
    document,
    config_digest(),
    max_bytes=65536,
    max_depth=8,
  )
}

///|
test "real Qwen3.5-MoE nested semantics produce an exact text-only profile" {
  let document = selected_qwen35_config()
  let from_string = parse_selected(document)
  let from_bytes = @qwen35_moe_config.parse_bytes(
    @utf8.encode(document),
    config_digest(),
    max_bytes=65536,
    max_depth=8,
  )
  assert_eq(from_string, from_bytes)
  assert_eq(from_string.content_digest(), config_digest())
  let spec = from_string.spec()
  assert_eq(spec.layer_count(), 40)
  assert_eq(spec.layer_kind(3), Some(FullAttention))
  assert_eq(spec.layer_kind(4), Some(GatedDeltaLinearAttention))
  assert_eq(spec.routed_expert_count(), 256)
  assert_eq(spec.experts_per_token(), 8)
  assert_eq(spec.matrix_dtype(), Fp8E4M3)
  assert_eq(spec.inverse_scale_dtype(), BFloat16)
  assert_false(spec.mtp_execution_enabled())
  assert_false(spec.vision_execution_enabled())
}

///|
test "dense Qwen and semantic drift never fall back into the MoE profile" {
  let source = selected_qwen35_config()
  let mutations : Array[(String, String, Qwen35ConfigError)] = [
    (
      "Qwen3_5MoeForConditionalGeneration",
      "Qwen3ForCausalLM",
      InvalidValue(
        area=Root,
        field=Architectures,
        issue=UnsupportedValue,
        index=0,
      ),
    ),
    (
      "\"model_type\":\"qwen3_5_moe_text\"",
      "\"model_type\":\"qwen3\"",
      InvalidValue(
        area=Text,
        field=TextModelType,
        issue=UnsupportedValue,
        index=0,
      ),
    ),
    (
      "\"full_attention_interval\":4",
      "\"full_attention_interval\":5",
      InvalidValue(
        area=Text,
        field=FullAttentionInterval,
        issue=UnsupportedValue,
        index=0,
      ),
    ),
    (
      "\"num_experts_per_tok\":8",
      "\"num_experts_per_tok\":7",
      InvalidValue(
        area=Text,
        field=ExpertsPerToken,
        issue=UnsupportedValue,
        index=0,
      ),
    ),
    (
      "\"mtp_num_hidden_layers\":1",
      "\"mtp_num_hidden_layers\":0",
      InvalidValue(
        area=Text,
        field=MtpLayerCount,
        issue=UnsupportedValue,
        index=0,
      ),
    ),
    (
      "\"fmt\":\"e4m3\"",
      "\"fmt\":\"e5m2\"",
      InvalidValue(
        area=Quantization,
        field=Fp8Format,
        issue=UnsupportedValue,
        index=0,
      ),
    ),
  ]
  for mutation in mutations {
    let (old, replacement, expected) = mutation
    try parse_selected(source.replace(old~, new=replacement)) catch {
      error => assert_eq(error, expected)
    } noraise {
      _ => fail("hostile Qwen3.5-MoE mutation was admitted")
    }
  }
}

///|
test "hybrid pattern, mRoPE, and block-FP8 geometry are exact" {
  let source = selected_qwen35_config()
  let mutations = [
    (
      "\"layer_types\":[\"linear_attention\"", "\"layer_types\":[\"full_attention\"",
    ),
    ("\"mrope_section\":[11,11,10]", "\"mrope_section\":[12,10,10]"),
    ("\"weight_block_size\":[128,128]", "\"weight_block_size\":[128,64]"),
    ("\"attn_output_gate\":true", "\"attn_output_gate\":false"),
  ]
  for mutation in mutations {
    let (old, replacement) = mutation
    try parse_selected(source.replace(old~, new=replacement)) catch {
      InvalidValue(issue=UnsupportedValue, ..) => ()
      _ => fail("unexpected exact-profile rejection")
    } noraise {
      _ => fail("hostile Qwen3.5-MoE geometry was admitted")
    }
  }
}

///|
test "nested duplicates, unknown text fields, and missing scale exclusions fail" {
  let source = selected_qwen35_config()
  let duplicate = source.replace(
    old="\"hidden_size\":2048,",
    new="\"hidden_size\":2048,\"hidden_size\":2048,",
  )
  try parse_selected(duplicate) catch {
    DuplicateField(depth=2, index=_) => ()
    _ => fail("unexpected duplicate rejection")
  } noraise {
    _ => fail("duplicate nested Qwen field was admitted")
  }
  let unknown = source.replace(
    old="\"use_cache\":true,",
    new="\"use_cache\":true,\"foreign_decoder\":true,",
  )
  try parse_selected(unknown) catch {
    UnknownField(area=Text, index=_) => ()
    _ => fail("unexpected unknown-field rejection")
  } noraise {
    _ => fail("foreign decoder field was admitted")
  }
  let missing_exclusion = source.replace(
    old="[\"model.embed_tokens\",\"lm_head\"]",
    new="[\"model.embed_tokens\"]",
  )
  try parse_selected(missing_exclusion) catch {
    InvalidValue(
      area=Quantization,
      field=ModulesNotConverted,
      issue=MissingRequiredEntry,
      index=0
    ) => ()
    _ => fail("unexpected module-exclusion rejection")
  } noraise {
    _ => fail("missing lm-head FP8 exclusion was admitted")
  }
}

///|
test "bounded reader rejects malformed, oversized, deep, and invalid UTF-8 input" {
  try parse_selected("{") catch {
    MalformedJson => ()
    _ => fail("unexpected malformed JSON rejection")
  } noraise {
    _ => fail("malformed Qwen config was admitted")
  }
  try
    @qwen35_moe_config.parse_string(
      selected_qwen35_config(),
      config_digest(),
      max_bytes=8,
      max_depth=8,
    )
  catch {
    ByteLimitExceeded(limit=8) => ()
    _ => fail("unexpected byte-limit rejection")
  } noraise {
    _ => fail("oversized Qwen config was admitted")
  }
  try
    @qwen35_moe_config.parse_string(
      selected_qwen35_config(),
      config_digest(),
      max_bytes=65536,
      max_depth=2,
    )
  catch {
    NestingLimitExceeded(limit=2) => ()
    _ => fail("unexpected nesting-limit rejection")
  } noraise {
    _ => fail("over-deep Qwen config was admitted")
  }
  try
    @qwen35_moe_config.parse_bytes(
      [0xff],
      config_digest(),
      max_bytes=1,
      max_depth=8,
    )
  catch {
    InvalidUtf8 => ()
    _ => fail("unexpected UTF-8 rejection")
  } noraise {
    _ => fail("invalid UTF-8 Qwen config was admitted")
  }
}
