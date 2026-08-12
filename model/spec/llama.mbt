///|
/// Validated dimensions and precision policy for a dense Llama-style decoder.
///
/// Construction is only available through `LlamaModelSpec::new`, so every
/// instance is safe for immutable model-plan construction.
pub struct LlamaModelSpec {
  vocabulary_size : Int
  hidden_size : Int
  intermediate_size : Int
  layer_count : Int
  attention_head_count : Int
  kv_head_count : Int
  head_dimension : Int
  context_length : Int
  rope_theta : Double
  rms_norm_epsilon : Double
  weight_dtype : ModelDType
  kv_dtype : ModelDType
} derive(Debug, Eq)

///|
/// Immutable plan-ready model metadata and its cache-salting identity.
pub struct LlamaModelMetadata {
  spec : LlamaModelSpec
  identity : ModelIdentity
} derive(Debug, Eq)

///|
/// Validates the v1 dense Llama architecture contract.
pub fn LlamaModelSpec::new(
  vocabulary_size~ : Int,
  hidden_size~ : Int,
  intermediate_size~ : Int,
  layer_count~ : Int,
  attention_head_count~ : Int,
  kv_head_count~ : Int,
  context_length~ : Int,
  rope_theta~ : Double,
  rms_norm_epsilon? : Double = 0.000001,
  weight_dtype~ : ModelDType,
  kv_dtype~ : ModelDType,
) -> LlamaModelSpec raise ModelSpecError {
  if vocabulary_size <= 0 {
    raise InvalidDimension(Vocabulary)
  }
  if hidden_size <= 0 {
    raise InvalidDimension(Hidden)
  }
  if intermediate_size <= 0 {
    raise InvalidDimension(Intermediate)
  }
  if layer_count <= 0 {
    raise InvalidDimension(LayerCount)
  }
  if attention_head_count <= 0 {
    raise InvalidDimension(AttentionHeadCount)
  }
  if kv_head_count <= 0 {
    raise InvalidDimension(KvHeadCount)
  }
  if context_length <= 0 {
    raise InvalidDimension(ContextLength)
  }
  if hidden_size % attention_head_count != 0 {
    raise InvalidHeadRelationship(HiddenNotDivisibleByAttentionHeads)
  }
  if kv_head_count > attention_head_count {
    raise InvalidHeadRelationship(KvHeadsExceedAttentionHeads)
  }
  if attention_head_count % kv_head_count != 0 {
    raise InvalidHeadRelationship(AttentionHeadsNotDivisibleByKvHeads)
  }
  if rope_theta.is_nan() || rope_theta <= 0.0 || rope_theta > 1.0e12 {
    raise InvalidRopeTheta
  }
  if rms_norm_epsilon.is_nan() ||
    rms_norm_epsilon.is_inf() ||
    rms_norm_epsilon <= 0.0 ||
    rms_norm_epsilon > 1.0 {
    raise InvalidRmsNormEpsilon
  }
  if weight_dtype != BFloat16 {
    raise UnsupportedWeightDType
  }
  if kv_dtype != BFloat16 {
    raise UnsupportedKvDType
  }
  {
    vocabulary_size,
    hidden_size,
    intermediate_size,
    layer_count,
    attention_head_count,
    kv_head_count,
    head_dimension: hidden_size / attention_head_count,
    context_length,
    rope_theta,
    rms_norm_epsilon,
    weight_dtype,
    kv_dtype,
  }
}

///|
pub fn LlamaModelSpec::vocabulary_size(self : LlamaModelSpec) -> Int {
  self.vocabulary_size
}

///|
pub fn LlamaModelSpec::hidden_size(self : LlamaModelSpec) -> Int {
  self.hidden_size
}

///|
pub fn LlamaModelSpec::intermediate_size(self : LlamaModelSpec) -> Int {
  self.intermediate_size
}

///|
pub fn LlamaModelSpec::layer_count(self : LlamaModelSpec) -> Int {
  self.layer_count
}

///|
pub fn LlamaModelSpec::attention_head_count(self : LlamaModelSpec) -> Int {
  self.attention_head_count
}

///|
pub fn LlamaModelSpec::kv_head_count(self : LlamaModelSpec) -> Int {
  self.kv_head_count
}

///|
pub fn LlamaModelSpec::head_dimension(self : LlamaModelSpec) -> Int {
  self.head_dimension
}

///|
pub fn LlamaModelSpec::context_length(self : LlamaModelSpec) -> Int {
  self.context_length
}

///|
pub fn LlamaModelSpec::rope_theta(self : LlamaModelSpec) -> Double {
  self.rope_theta
}

///|
pub fn LlamaModelSpec::rms_norm_epsilon(self : LlamaModelSpec) -> Double {
  self.rms_norm_epsilon
}

///|
pub fn LlamaModelSpec::weight_dtype(self : LlamaModelSpec) -> ModelDType {
  self.weight_dtype
}

///|
pub fn LlamaModelSpec::kv_dtype(self : LlamaModelSpec) -> ModelDType {
  self.kv_dtype
}

///|
pub fn LlamaModelMetadata::new(
  spec : LlamaModelSpec,
  identity : ModelIdentity,
) -> LlamaModelMetadata {
  { spec, identity }
}

///|
pub fn LlamaModelMetadata::spec(self : LlamaModelMetadata) -> LlamaModelSpec {
  self.spec
}

///|
pub fn LlamaModelMetadata::identity(self : LlamaModelMetadata) -> ModelIdentity {
  self.identity
}
