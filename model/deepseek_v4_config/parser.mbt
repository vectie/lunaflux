///|
/// Bounded, payload-safe failures from exact official DeepSeek-V4 config
/// admission. Field indices refer to the package's stable closed schema; no
/// attacker-controlled key or value is retained in an error.
pub(all) suberror DeepSeekV4ConfigError {
  InvalidByteLimit
  InvalidDepthLimit
  ByteLimitExceeded(Int)
  InvalidUtf8
  MalformedJson
  NonObjectRoot
  NestingLimitExceeded(Int)
  DuplicateField(Int, Int)
  UnknownField(Int)
  ExecutableMetadata(Int)
  MissingField(Int)
  InvalidField(Int)
  UnsupportedProfile
} derive(Debug, Eq)

///|
let schema_fields : ReadOnlyArray[String] = [
  "architectures", "attention_bias", "attention_dropout", "bos_token_id", "compress_ratios",
  "compress_rope_theta", "eos_token_id", "expert_dtype", "hc_eps", "hc_mult", "hc_sinkhorn_iters",
  "head_dim", "hidden_act", "hidden_size", "index_head_dim", "index_n_heads", "index_topk",
  "initializer_range", "max_position_embeddings", "model_type", "moe_intermediate_size",
  "n_routed_experts", "n_shared_experts", "norm_topk_prob", "num_attention_heads",
  "num_experts_per_tok", "num_hash_layers", "num_hidden_layers", "num_key_value_heads",
  "num_nextn_predict_layers", "o_groups", "o_lora_rank", "q_lora_rank", "qk_rope_head_dim",
  "quantization_config", "rms_norm_eps", "rope_scaling", "rope_theta", "routed_scaling_factor",
  "scoring_func", "sliding_window", "swiglu_limit", "tie_word_embeddings", "topk_method",
  "torch_dtype", "transformers_version", "use_cache", "vocab_size", "dspark_block_size",
  "dspark_markov_rank", "dspark_noise_token_id", "dspark_target_layer_ids",
]

///|
fn field_index(name : StringView) -> Int? {
  for index, field in schema_fields {
    if field[:] == name {
      return Some(index)
    }
  }
  None
}

///|
fn required(
  root : Map[String, Json],
  name : String,
) -> Json raise DeepSeekV4ConfigError {
  let index = field_index(name).unwrap()
  root.get(name).unwrap_or_else(() => raise MissingField(index))
}

///|
fn exact_int(value : Json, index : Int) -> Int raise DeepSeekV4ConfigError {
  guard value is Number(number, repr=_) &&
    !number.is_nan() &&
    !number.is_inf() &&
    number == number.trunc() &&
    number >= -2147483648.0 &&
    number <= 2147483647.0 else {
    raise InvalidField(index)
  }
  number.to_int()
}

///|
fn expect_int(
  root : Map[String, Json],
  name : String,
  expected : Int,
) -> Unit raise DeepSeekV4ConfigError {
  let index = field_index(name).unwrap()
  if exact_int(required(root, name), index) != expected {
    raise InvalidField(index)
  }
}

///|
fn expect_number(
  root : Map[String, Json],
  name : String,
  expected : Double,
) -> Unit raise DeepSeekV4ConfigError {
  let index = field_index(name).unwrap()
  match required(root, name) {
    Number(value, repr=_) if !value.is_nan() &&
      !value.is_inf() &&
      value == expected => ()
    _ => raise InvalidField(index)
  }
}

///|
fn expect_string(
  root : Map[String, Json],
  name : String,
  expected : String,
) -> Unit raise DeepSeekV4ConfigError {
  let index = field_index(name).unwrap()
  match required(root, name) {
    String(value) if value == expected => ()
    _ => raise InvalidField(index)
  }
}

///|
fn expect_bool(
  root : Map[String, Json],
  name : String,
  expected : Bool,
) -> Unit raise DeepSeekV4ConfigError {
  let index = field_index(name).unwrap()
  match required(root, name) {
    True if expected => ()
    False if !expected => ()
    _ => raise InvalidField(index)
  }
}

///|
fn expect_int_array(
  root : Map[String, Json],
  name : String,
  expected : ReadOnlyArray[Int],
) -> Unit raise DeepSeekV4ConfigError {
  let index = field_index(name).unwrap()
  guard required(root, name) is Array(values) &&
    values.length() == expected.length() else {
    raise InvalidField(index)
  }
  for position, item in values {
    if exact_int(item, index) != expected[position] {
      raise InvalidField(index)
    }
  }
}

///|
fn expect_object_keys(
  object : Map[String, Json],
  names : ReadOnlyArray[String],
  parent_index : Int,
) -> Unit raise DeepSeekV4ConfigError {
  if object.length() != names.length() {
    raise InvalidField(parent_index)
  }
  for key, _ in object {
    if !names.any(name => name == key) {
      raise InvalidField(parent_index)
    }
  }
}

///|
fn nested_required(
  object : Map[String, Json],
  name : String,
  parent : Int,
) -> Json raise DeepSeekV4ConfigError {
  object.get(name).unwrap_or_else(() => raise InvalidField(parent))
}

///|
fn nested_number(
  object : Map[String, Json],
  name : String,
  expected : Double,
  parent : Int,
) -> Unit raise DeepSeekV4ConfigError {
  match nested_required(object, name, parent) {
    Number(value, repr=_) if !value.is_nan() &&
      !value.is_inf() &&
      value == expected => ()
    _ => raise InvalidField(parent)
  }
}

///|
fn nested_string(
  object : Map[String, Json],
  name : String,
  expected : String,
  parent : Int,
) -> Unit raise DeepSeekV4ConfigError {
  match nested_required(object, name, parent) {
    String(value) if value == expected => ()
    _ => raise InvalidField(parent)
  }
}

///|
fn nested_int(
  object : Map[String, Json],
  name : String,
  expected : Int,
  parent : Int,
) -> Unit raise DeepSeekV4ConfigError {
  if exact_int(nested_required(object, name, parent), parent) != expected {
    raise InvalidField(parent)
  }
}

///|
fn nested_int_array(
  object : Map[String, Json],
  name : String,
  expected : ReadOnlyArray[Int],
  parent : Int,
) -> Unit raise DeepSeekV4ConfigError {
  guard nested_required(object, name, parent) is Array(values) &&
    values.length() == expected.length() else {
    raise InvalidField(parent)
  }
  for index, value in values {
    if exact_int(value, parent) != expected[index] {
      raise InvalidField(parent)
    }
  }
}

///|
fn validate_nested(
  root : Map[String, Json],
) -> Unit raise DeepSeekV4ConfigError {
  let rope_index = field_index("rope_scaling").unwrap()
  guard required(root, "rope_scaling") is Object(rope) else {
    raise InvalidField(rope_index)
  }
  expect_object_keys(
    rope,
    [
      "beta_fast", "beta_slow", "factor", "original_max_position_embeddings", "type",
    ],
    rope_index,
  )
  nested_number(rope, "beta_fast", 32.0, rope_index)
  nested_number(rope, "beta_slow", 1.0, rope_index)
  nested_number(rope, "factor", 16.0, rope_index)
  nested_int(rope, "original_max_position_embeddings", 65536, rope_index)
  nested_string(rope, "type", "yarn", rope_index)

  let quant_index = field_index("quantization_config").unwrap()
  guard required(root, "quantization_config") is Object(quant) else {
    raise InvalidField(quant_index)
  }
  expect_object_keys(
    quant,
    [
      "activation_scheme", "fmt", "quant_method", "scale_fmt", "weight_block_size",
    ],
    quant_index,
  )
  nested_string(quant, "activation_scheme", "dynamic", quant_index)
  nested_string(quant, "fmt", "e4m3", quant_index)
  nested_string(quant, "quant_method", "fp8", quant_index)
  nested_string(quant, "scale_fmt", "ue8m0", quant_index)
  nested_int_array(quant, "weight_block_size", [128, 128], quant_index)
}

///|
fn select_spec(
  root : Map[String, Json],
) -> @deepseek_v4_spec.DeepSeekV4ModelSpec raise DeepSeekV4ConfigError {
  let hidden = exact_int(
    required(root, "hidden_size"),
    field_index("hidden_size").unwrap(),
  )
  let expert = match required(root, "expert_dtype") {
    String(value) => value
    _ => raise InvalidField(field_index("expert_dtype").unwrap())
  }
  let dspark = root.contains("dspark_block_size")
  let model_id = match (hidden, expert, dspark) {
    (4096, "fp4", false) => "deepseek-ai/DeepSeek-V4-Flash"
    (4096, "fp8", false) => "deepseek-ai/DeepSeek-V4-Flash-Base"
    (7168, "fp4", false) => "deepseek-ai/DeepSeek-V4-Pro"
    (7168, "fp8", false) => "deepseek-ai/DeepSeek-V4-Pro-Base"
    (4096, "fp4", true) => "deepseek-ai/DeepSeek-V4-Flash-0731"
    _ => raise UnsupportedProfile
  }
  @deepseek_v4_spec.DeepSeekV4ModelSpec::from_official_model_id(model_id) catch {
    _ => raise UnsupportedProfile
  }
}

///|
fn validate_exact(
  root : Map[String, Json],
  spec : @deepseek_v4_spec.DeepSeekV4ModelSpec,
) -> Unit raise DeepSeekV4ConfigError {
  let mut unknown_index = 0
  for key, _ in root {
    match key {
      "auto_map" => raise ExecutableMetadata(0)
      "trust_remote_code" => raise ExecutableMetadata(1)
      "custom_pipelines" => raise ExecutableMetadata(2)
      _ => if field_index(key) is None { raise UnknownField(unknown_index) }
    }
    unknown_index += 1
  }
  let required_count = if spec.has_dspark() { 52 } else { 48 }
  if root.length() != required_count {
    for index in 0..<required_count {
      if root.get(schema_fields[index]) is None {
        raise MissingField(index)
      }
    }
    raise UnknownField(root.length())
  }
  guard required(root, "architectures") is Array(architectures) &&
    architectures.length() == 1 else {
    raise InvalidField(field_index("architectures").unwrap())
  }
  match architectures[0] {
    String(value) if value == spec.architecture() => ()
    _ => raise InvalidField(field_index("architectures").unwrap())
  }
  expect_string(root, "model_type", spec.model_type())
  expect_int(root, "vocab_size", spec.vocabulary_size())
  expect_int(root, "hidden_size", spec.hidden_size())
  expect_int(root, "moe_intermediate_size", spec.moe_intermediate_size())
  expect_int(root, "num_hidden_layers", spec.layer_count())
  expect_int(root, "num_attention_heads", spec.attention_head_count())
  expect_int(root, "num_key_value_heads", spec.kv_head_count())
  expect_int(root, "head_dim", spec.head_dimension())
  expect_int(root, "qk_rope_head_dim", spec.rope_head_dimension())
  expect_int(root, "q_lora_rank", spec.q_lora_rank())
  expect_int(root, "o_lora_rank", spec.o_lora_rank())
  expect_int(root, "o_groups", spec.output_group_count())
  expect_int(root, "n_routed_experts", spec.routed_expert_count())
  expect_int(root, "n_shared_experts", spec.shared_expert_count())
  expect_int(root, "num_experts_per_tok", spec.experts_per_token())
  expect_int(root, "num_hash_layers", spec.hash_layer_count())
  expect_int(
    root,
    "num_nextn_predict_layers",
    spec.next_token_prediction_layer_count(),
  )
  expect_int(root, "max_position_embeddings", spec.max_position_embeddings())
  expect_int(root, "sliding_window", spec.sliding_window())
  expect_int(root, "compress_rope_theta", spec.compressed_rope_theta().to_int())
  expect_int(root, "rope_theta", spec.rope_theta().to_int())
  expect_int(root, "hc_mult", spec.hyper_connection_width())
  expect_int(
    root,
    "hc_sinkhorn_iters",
    spec.hyper_connection_sinkhorn_iterations(),
  )
  expect_int(root, "index_head_dim", spec.index_head_dimension())
  expect_int(root, "index_n_heads", spec.index_head_count())
  expect_int(root, "index_topk", spec.index_top_k())
  expect_int(root, "bos_token_id", 0)
  expect_int(root, "eos_token_id", 1)
  expect_number(root, "routed_scaling_factor", spec.routed_scaling_factor())
  expect_number(root, "rms_norm_eps", spec.rms_norm_epsilon())
  expect_number(root, "hc_eps", spec.hyper_connection_epsilon())
  expect_number(root, "initializer_range", 0.02)
  expect_number(root, "attention_dropout", 0.0)
  expect_number(root, "swiglu_limit", spec.swiglu_limit())
  expect_string(
    root,
    "expert_dtype",
    if spec.expert_storage() is Fp4E2M1 {
      "fp4"
    } else {
      "fp8"
    },
  )
  expect_string(root, "hidden_act", "silu")
  expect_string(root, "scoring_func", "sqrtsoftplus")
  expect_string(root, "topk_method", "noaux_tc")
  expect_string(root, "torch_dtype", "bfloat16")
  expect_string(root, "transformers_version", "4.57.1")
  expect_bool(root, "attention_bias", false)
  expect_bool(root, "norm_topk_prob", true)
  expect_bool(root, "tie_word_embeddings", false)
  expect_bool(root, "use_cache", true)
  let ratios : Array[Int] = []
  for index in 0..<spec.attention_profile_count() {
    ratios.push(spec.compression_ratio(index).unwrap())
  }
  expect_int_array(root, "compress_ratios", ReadOnlyArray::from_array(ratios))
  validate_nested(root)
  if spec.has_dspark() {
    expect_int(root, "dspark_block_size", spec.dspark_block_size())
    expect_int(root, "dspark_markov_rank", spec.dspark_markov_rank())
    expect_int(root, "dspark_noise_token_id", spec.dspark_noise_token_id())
    expect_int_array(
      root,
      "dspark_target_layer_ids",
      spec.dspark_target_layer_ids(),
    )
  }
}

///|
fn parse_text(
  document : StringView,
  content : @spec.ContentDigest,
  max_depth : Int,
) -> @deepseek_v4_spec.DeepSeekV4ModelMetadata raise DeepSeekV4ConfigError {
  @json_guard.reject_duplicate_object_keys(document, max_depth~) catch {
    InvalidDepthLimit => raise InvalidDepthLimit
    DepthLimitExceeded(limit~) => raise NestingLimitExceeded(limit)
    DuplicateObjectKey(depth~, index~) => raise DuplicateField(depth, index)
    MalformedStructure => raise MalformedJson
  }
  let parsed = @json.parse(document) catch { _ => raise MalformedJson }
  guard parsed is Object(root) else { raise NonObjectRoot }
  let spec = select_spec(root)
  validate_exact(root, spec)
  @deepseek_v4_spec.DeepSeekV4ModelMetadata::from_verified_content(
    spec, content,
  )
}

///|
pub fn parse_bytes(
  document : BytesView,
  content : @spec.ContentDigest,
  max_bytes~ : Int,
  max_depth~ : Int,
) -> @deepseek_v4_spec.DeepSeekV4ModelMetadata raise DeepSeekV4ConfigError {
  if max_bytes <= 0 {
    raise InvalidByteLimit
  }
  if max_depth <= 0 {
    raise InvalidDepthLimit
  }
  if document.length() > max_bytes {
    raise ByteLimitExceeded(max_bytes)
  }
  let text = @utf8.decode(document) catch { _ => raise InvalidUtf8 }
  parse_text(text, content, max_depth)
}

///|
pub fn parse_string(
  document : StringView,
  content : @spec.ContentDigest,
  max_bytes~ : Int,
  max_depth~ : Int,
) -> @deepseek_v4_spec.DeepSeekV4ModelMetadata raise DeepSeekV4ConfigError {
  if max_bytes <= 0 {
    raise InvalidByteLimit
  }
  if max_depth <= 0 {
    raise InvalidDepthLimit
  }
  let encoded = @utf8.encode(document)
  if encoded.length() > max_bytes {
    raise ByteLimitExceeded(max_bytes)
  }
  parse_text(document, content, max_depth)
}
