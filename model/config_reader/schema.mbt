///|
fn executable_field(name : StringView) -> ExecutableField? {
  match name {
    "auto_map" => Some(AutoMap)
    "trust_remote_code" => Some(TrustRemoteCode)
    "custom_pipelines" => Some(CustomPipelines)
    _ => None
  }
}

///|
fn is_semantic_field(name : StringView) -> Bool {
  match name {
    "architectures"
    | "model_type"
    | "vocab_size"
    | "hidden_size"
    | "intermediate_size"
    | "num_hidden_layers"
    | "num_attention_heads"
    | "num_key_value_heads"
    | "max_position_embeddings"
    | "rope_theta"
    | "torch_dtype"
    | "tie_word_embeddings"
    | "rope_scaling"
    | "sliding_window"
    | "hidden_act"
    | "rms_norm_eps"
    | "attention_bias"
    | "attention_dropout"
    | "mlp_bias"
    | "head_dim"
    | "pretraining_tp" => true
    _ => false
  }
}

///|
fn is_inert_field(name : StringView) -> Bool {
  match name {
    "_name_or_path"
    | "transformers_version"
    | "bos_token_id"
    | "eos_token_id"
    | "pad_token_id"
    | "initializer_range"
    | "use_cache" => true
    _ => false
  }
}

///|
fn validate_known_fields(
  object : Map[String, Json],
) -> Unit raise ModelConfigReaderError {
  let mut index = 0
  for name, _ in object {
    match executable_field(name) {
      Some(field) => raise ExecutableMetadata(field)
      None =>
        if !is_semantic_field(name) && !is_inert_field(name) {
          raise UnknownField(index~)
        }
    }
    index += 1
  }
}

///|
fn is_qwen3_field(name : StringView) -> Bool {
  is_semantic_field(name) ||
  is_inert_field(name) ||
  name == "max_window_layers" ||
  name == "use_sliding_window"
}

///|
fn validate_qwen3_known_fields(
  object : Map[String, Json],
) -> Unit raise ModelConfigReaderError {
  let mut index = 0
  for name, _ in object {
    match executable_field(name) {
      Some(field) => raise ExecutableMetadata(field)
      None => if !is_qwen3_field(name) { raise UnknownField(index~) }
    }
    index += 1
  }
}

///|
fn check_optional_inert_string(
  object : Map[String, Json],
  name : String,
) -> Unit raise ModelConfigReaderError {
  match object.get(name) {
    None | Some(String(_)) => ()
    Some(_) => raise invalid(InertMetadata, ExpectedString)
  }
}

///|
fn check_optional_inert_token_id(
  object : Map[String, Json],
  name : String,
) -> Unit raise ModelConfigReaderError {
  match object.get(name) {
    None | Some(Null) => ()
    Some(value) => {
      let token = exact_int64(value, InertMetadata)
      if token < 0L || token > 2147483647L {
        raise invalid(InertMetadata, ExpectedInteger)
      }
    }
  }
}

///|
fn validate_inert_fields(
  object : Map[String, Json],
) -> Unit raise ModelConfigReaderError {
  check_optional_inert_string(object, "_name_or_path")
  check_optional_inert_string(object, "transformers_version")
  check_optional_inert_token_id(object, "bos_token_id")
  check_optional_inert_token_id(object, "eos_token_id")
  check_optional_inert_token_id(object, "pad_token_id")
  match object.get("initializer_range") {
    None | Some(Number(_, repr=_)) => ()
    Some(_) => raise invalid(InertMetadata, ExpectedNumber)
  }
  match object.get("use_cache") {
    None | Some(True | False) => ()
    Some(_) => raise invalid(InertMetadata, ExpectedBoolean)
  }
}

///|
fn validate_llama_semantics(
  object : Map[String, Json],
  hidden_size : Int,
  attention_heads : Int,
) -> Unit raise ModelConfigReaderError {
  require_single_architecture(object, "LlamaForCausalLM")
  if require_string(object, "model_type", ModelType) != "llama" {
    raise invalid(ModelType, UnsupportedValue)
  }
  if require_bool(object, "tie_word_embeddings", TieWordEmbeddings) {
    raise invalid(TieWordEmbeddings, UnsupportedValue)
  }
  check_optional_null(object, "rope_scaling", RopeScaling)
  check_optional_null(object, "sliding_window", SlidingWindow)
  check_optional_string(object, "hidden_act", HiddenActivation, "silu")
  check_optional_bool(object, "attention_bias", AttentionBias, false)
  check_optional_number(object, "attention_dropout", AttentionDropout, 0.0)
  check_optional_bool(object, "mlp_bias", MlpBias, false)
  check_optional_int(
    object,
    "head_dim",
    HeadDimension,
    hidden_size / attention_heads,
  )
  check_optional_int(object, "pretraining_tp", PretrainingTensorParallelism, 1)
}

///|
fn decode_model(
  object : Map[String, Json],
  content_digest : @spec.ContentDigest,
) -> @spec.LlamaModelMetadata raise ModelConfigReaderError {
  validate_known_fields(object)
  validate_inert_fields(object)
  let vocabulary_size = require_int(object, "vocab_size", VocabularySize)
  let hidden_size = require_int(object, "hidden_size", HiddenSize)
  let intermediate_size = require_int(
    object,
    "intermediate_size",
    IntermediateSize,
  )
  let layer_count = require_int(object, "num_hidden_layers", LayerCount)
  let attention_head_count = require_int(
    object,
    "num_attention_heads",
    AttentionHeadCount,
  )
  let kv_head_count = match
    optional_int(object, "num_key_value_heads", KvHeadCount) {
    Some(value) => value
    None => attention_head_count
  }
  let context_length = require_int(
    object,
    "max_position_embeddings",
    ContextLength,
  )
  let rope_theta = require_number(object, "rope_theta", RopeTheta)
  let rms_norm_epsilon = optional_finite_number(
    object,
    "rms_norm_eps",
    RmsNormEpsilon,
    0.000001,
  )
  let dtype = require_string(object, "torch_dtype", TorchDType)
  if dtype != "bfloat16" && dtype != "BF16" {
    raise invalid(TorchDType, UnsupportedValue)
  }
  let model_spec = @spec.LlamaModelSpec::new(
    vocabulary_size~,
    hidden_size~,
    intermediate_size~,
    layer_count~,
    attention_head_count~,
    kv_head_count~,
    context_length~,
    rope_theta~,
    rms_norm_epsilon~,
    weight_dtype=BFloat16,
    kv_dtype=BFloat16,
  ) catch {
    error => raise Validation(error)
  }
  validate_llama_semantics(object, hidden_size, attention_head_count)
  @spec.LlamaModelMetadata::from_verified_content(model_spec, content_digest) catch {
    error => raise Validation(error)
  }
}

///|
fn validate_mistral_semantics(
  object : Map[String, Json],
  hidden_size : Int,
  attention_heads : Int,
) -> Int? raise ModelConfigReaderError {
  require_single_architecture(object, "MistralForCausalLM")
  if require_string(object, "model_type", ModelType) != "mistral" {
    raise invalid(ModelType, UnsupportedValue)
  }
  if require_bool(object, "tie_word_embeddings", TieWordEmbeddings) {
    raise invalid(TieWordEmbeddings, UnsupportedValue)
  }
  check_optional_null(object, "rope_scaling", RopeScaling)
  check_optional_string(object, "hidden_act", HiddenActivation, "silu")
  check_optional_bool(object, "attention_bias", AttentionBias, false)
  check_optional_number(object, "attention_dropout", AttentionDropout, 0.0)
  check_optional_bool(object, "mlp_bias", MlpBias, false)
  check_optional_int(
    object,
    "head_dim",
    HeadDimension,
    hidden_size / attention_heads,
  )
  check_optional_int(object, "pretraining_tp", PretrainingTensorParallelism, 1)
  optional_nullable_int(object, "sliding_window", SlidingWindow)
}

///|
fn decode_mistral_model(
  object : Map[String, Json],
  content_digest : @spec.ContentDigest,
) -> @mistral_spec.MistralModelMetadata raise ModelConfigReaderError {
  validate_known_fields(object)
  validate_inert_fields(object)
  let vocabulary_size = require_int(object, "vocab_size", VocabularySize)
  let hidden_size = require_int(object, "hidden_size", HiddenSize)
  let intermediate_size = require_int(
    object,
    "intermediate_size",
    IntermediateSize,
  )
  let layer_count = require_int(object, "num_hidden_layers", LayerCount)
  let attention_head_count = require_int(
    object,
    "num_attention_heads",
    AttentionHeadCount,
  )
  let kv_head_count = match
    optional_int(object, "num_key_value_heads", KvHeadCount) {
    Some(value) => value
    None => attention_head_count
  }
  let max_position_embeddings = require_int(
    object,
    "max_position_embeddings",
    ContextLength,
  )
  let rope_theta = require_number(object, "rope_theta", RopeTheta)
  let rms_norm_epsilon = optional_finite_number(
    object,
    "rms_norm_eps",
    RmsNormEpsilon,
    0.000001,
  )
  let dtype = require_string(object, "torch_dtype", TorchDType)
  if dtype != "bfloat16" && dtype != "BF16" {
    raise invalid(TorchDType, UnsupportedValue)
  }
  let sliding_window = validate_mistral_semantics(
    object, hidden_size, attention_head_count,
  )
  let model_spec = @mistral_spec.MistralModelSpec::new(
    vocabulary_size~,
    hidden_size~,
    intermediate_size~,
    layer_count~,
    attention_head_count~,
    kv_head_count~,
    max_position_embeddings~,
    sliding_window?,
    rope_theta~,
    rms_norm_epsilon~,
    weight_dtype=BFloat16,
    kv_dtype=BFloat16,
  ) catch {
    error => raise Validation(error)
  }
  @mistral_spec.MistralModelMetadata::from_verified_content(
    model_spec, content_digest,
  ) catch {
    error => raise Validation(error)
  }
}

///|
fn require_pinned_qwen3_int(
  value : Int,
  expected : Int,
  field : ModelConfigField,
) -> Unit raise ModelConfigReaderError {
  if value != expected {
    raise invalid(field, UnsupportedValue)
  }
}

///|
fn decode_qwen3_model(
  object : Map[String, Json],
  content_digest : @spec.ContentDigest,
) -> @qwen3_spec.Qwen3ModelMetadata raise ModelConfigReaderError {
  validate_qwen3_known_fields(object)
  validate_inert_fields(object)
  require_single_architecture(object, "Qwen3ForCausalLM")
  if require_string(object, "model_type", ModelType) != "qwen3" {
    raise invalid(ModelType, UnsupportedValue)
  }
  let vocabulary_size = require_int(object, "vocab_size", VocabularySize)
  let hidden_size = require_int(object, "hidden_size", HiddenSize)
  let intermediate_size = require_int(
    object,
    "intermediate_size",
    IntermediateSize,
  )
  let layer_count = require_int(object, "num_hidden_layers", LayerCount)
  let attention_head_count = require_int(
    object,
    "num_attention_heads",
    AttentionHeadCount,
  )
  let kv_head_count = require_int(object, "num_key_value_heads", KvHeadCount)
  let head_dimension = require_int(object, "head_dim", HeadDimension)
  let context_length = require_int(
    object,
    "max_position_embeddings",
    ContextLength,
  )
  let rope_theta = require_number(object, "rope_theta", RopeTheta)
  let rms_norm_epsilon = require_number(object, "rms_norm_eps", RmsNormEpsilon)
  require_pinned_qwen3_int(vocabulary_size, 151936, VocabularySize)
  require_pinned_qwen3_int(hidden_size, 1024, HiddenSize)
  require_pinned_qwen3_int(intermediate_size, 3072, IntermediateSize)
  require_pinned_qwen3_int(layer_count, 28, LayerCount)
  require_pinned_qwen3_int(attention_head_count, 16, AttentionHeadCount)
  require_pinned_qwen3_int(kv_head_count, 8, KvHeadCount)
  require_pinned_qwen3_int(head_dimension, 128, HeadDimension)
  require_pinned_qwen3_int(context_length, 40960, ContextLength)
  if rope_theta != 1000000.0 {
    raise invalid(RopeTheta, UnsupportedValue)
  }
  if rms_norm_epsilon != 0.000001 {
    raise invalid(RmsNormEpsilon, UnsupportedValue)
  }
  let dtype = require_string(object, "torch_dtype", TorchDType)
  if dtype != "bfloat16" && dtype != "BF16" {
    raise invalid(TorchDType, UnsupportedValue)
  }
  if !require_bool(object, "tie_word_embeddings", TieWordEmbeddings) {
    raise invalid(TieWordEmbeddings, UnsupportedValue)
  }
  check_optional_null(object, "rope_scaling", RopeScaling)
  check_optional_null(object, "sliding_window", SlidingWindow)
  check_optional_string(object, "hidden_act", HiddenActivation, "silu")
  check_optional_bool(object, "attention_bias", AttentionBias, false)
  check_optional_number(object, "attention_dropout", AttentionDropout, 0.0)
  check_optional_bool(object, "mlp_bias", MlpBias, false)
  check_optional_int(object, "pretraining_tp", PretrainingTensorParallelism, 1)
  if require_int(object, "max_window_layers", MaxWindowLayers) != layer_count {
    raise invalid(MaxWindowLayers, UnsupportedValue)
  }
  if require_bool(object, "use_sliding_window", UseSlidingWindow) {
    raise invalid(UseSlidingWindow, UnsupportedValue)
  }
  let model_spec = @qwen3_spec.Qwen3ModelSpec::new(
    vocabulary_size~,
    hidden_size~,
    intermediate_size~,
    layer_count~,
    attention_head_count~,
    kv_head_count~,
    head_dimension~,
    context_length~,
    rope_theta~,
    rms_norm_epsilon~,
    weight_dtype=BFloat16,
    kv_dtype=BFloat16,
    tied_embeddings=true,
  ) catch {
    error => raise Qwen3Validation(error)
  }
  @qwen3_spec.Qwen3ModelMetadata::from_verified_content(
    model_spec, content_digest,
  ) catch {
    error => raise Qwen3Validation(error)
  }
}
