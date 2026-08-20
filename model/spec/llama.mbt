///|
/// Validated dimensions and precision policy for a dense Llama-style decoder.
///
/// Construction is only available through `LlamaModelSpec::new`, so every
/// instance is safe for immutable model-plan construction.
pub struct LlamaModelSpec {
  priv vocabulary_size : Int
  priv hidden_size : Int
  priv intermediate_size : Int
  priv layer_count : Int
  priv attention_head_count : Int
  priv kv_head_count : Int
  priv head_dimension : Int
  priv context_length : Int
  priv rope_theta : Double
  priv rms_norm_epsilon : Double
  priv weight_dtype : ModelDType
  priv kv_dtype : ModelDType
} derive(Debug, Eq)

///|
/// Immutable plan-ready model metadata and its cache-salting identity.
pub struct LlamaModelMetadata {
  priv spec : LlamaModelSpec
  priv identity : ModelIdentity
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
  let spec : LlamaModelSpec = {
    vocabulary_size,
    hidden_size,
    intermediate_size,
    layer_count,
    attention_head_count,
    kv_head_count,
    head_dimension: if hidden_size > 0 && attention_head_count > 0 {
      hidden_size / attention_head_count
    } else {
      0
    },
    context_length,
    rope_theta,
    rms_norm_epsilon,
    weight_dtype,
    kv_dtype,
  }
  spec.validate()
  spec
}

///|
fn LlamaModelSpec::validate(self : LlamaModelSpec) -> Unit raise ModelSpecError {
  if self.vocabulary_size <= 0 {
    raise InvalidDimension(Vocabulary)
  }
  if self.hidden_size <= 0 {
    raise InvalidDimension(Hidden)
  }
  if self.intermediate_size <= 0 {
    raise InvalidDimension(Intermediate)
  }
  if self.layer_count <= 0 {
    raise InvalidDimension(LayerCount)
  }
  if self.attention_head_count <= 0 {
    raise InvalidDimension(AttentionHeadCount)
  }
  if self.kv_head_count <= 0 {
    raise InvalidDimension(KvHeadCount)
  }
  if self.context_length <= 0 {
    raise InvalidDimension(ContextLength)
  }
  if self.hidden_size % self.attention_head_count != 0 {
    raise InvalidHeadRelationship(HiddenNotDivisibleByAttentionHeads)
  }
  if self.kv_head_count > self.attention_head_count {
    raise InvalidHeadRelationship(KvHeadsExceedAttentionHeads)
  }
  if self.attention_head_count % self.kv_head_count != 0 {
    raise InvalidHeadRelationship(AttentionHeadsNotDivisibleByKvHeads)
  }
  if self.head_dimension != self.hidden_size / self.attention_head_count {
    raise InvalidHeadRelationship(HeadDimensionMismatch)
  }
  if self.rope_theta.is_nan() ||
    self.rope_theta <= 0.0 ||
    self.rope_theta > 1.0e12 {
    raise InvalidRopeTheta
  }
  if self.rms_norm_epsilon.is_nan() ||
    self.rms_norm_epsilon.is_inf() ||
    self.rms_norm_epsilon <= 0.0 ||
    self.rms_norm_epsilon > 1.0 {
    raise InvalidRmsNormEpsilon
  }
  if self.weight_dtype != BFloat16 {
    raise UnsupportedWeightDType
  }
  if self.kv_dtype != BFloat16 {
    raise UnsupportedKvDType
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
/// Constructs plan-ready metadata from independently verified model content.
///
/// This admission constructor does not accept a caller-supplied plan identity.
/// It derives the plan digest from the complete validated semantic
/// specification.
pub fn LlamaModelMetadata::from_verified_content(
  spec : LlamaModelSpec,
  content : ContentDigest,
) -> LlamaModelMetadata raise ModelSpecError {
  spec.validate()
  content.validate()
  let metadata : LlamaModelMetadata = {
    spec,
    identity: ModelIdentity::new(content, spec.canonical_plan_digest()),
  }
  metadata.validate()
  metadata
}

///|
pub fn LlamaModelMetadata::spec(self : LlamaModelMetadata) -> LlamaModelSpec {
  self.spec
}

///|
pub fn LlamaModelMetadata::identity(self : LlamaModelMetadata) -> ModelIdentity {
  self.identity
}

///|
fn LlamaModelMetadata::validate(
  self : LlamaModelMetadata,
) -> Unit raise ModelSpecError {
  self.spec.validate()
  self.identity.validate()
  if self.identity.plan() != self.spec.canonical_plan_digest() {
    raise MetadataIdentityMismatch
  }
}
