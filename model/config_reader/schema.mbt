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
fn validate_selected_semantics(
  object : Map[String, Json],
  hidden_size : Int,
  attention_heads : Int,
) -> Unit raise ModelConfigReaderError {
  require_single_architecture(object)
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
  validate_selected_semantics(object, hidden_size, attention_head_count)
  @spec.LlamaModelMetadata::from_verified_content(model_spec, content_digest) catch {
    error => raise Validation(error)
  }
}
