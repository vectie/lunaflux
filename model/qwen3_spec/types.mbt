///|
/// Dimension rejected by the dense Qwen3 semantic profile.
pub(all) enum Qwen3ModelDimension {
  Vocabulary
  Hidden
  Intermediate
  LayerCount
  AttentionHeadCount
  KvHeadCount
  HeadDimension
  ContextLength
} derive(Debug, Eq)

///|
/// Relationship rejected by the dense Qwen3 semantic profile.
pub(all) enum Qwen3HeadRelationship {
  AttentionHeadsNotDivisibleByKvHeads
  KvHeadsExceedAttentionHeads
  OddHeadDimension
  ProjectionWidthOverflow
} derive(Debug, Eq)

///|
/// Payload-safe dense Qwen3 metadata validation failures.
pub(all) suberror Qwen3ModelSpecError {
  InvalidDimension(Qwen3ModelDimension)
  InvalidHeadRelationship(Qwen3HeadRelationship)
  InvalidRopeTheta
  InvalidRmsNormEpsilon
  UnsupportedWeightDType
  UnsupportedKvDType
  UntiedEmbeddings
} derive(Debug, Eq)

///|
/// Validated dimensions and precision policy for dense Qwen3.
///
/// Qwen3 deliberately owns an explicit attention head dimension. In the
/// pinned 0.6B model it is 128 while `hidden_size / attention_heads` is 64;
/// treating this as the older dense-Llama geometry would silently change the
/// query projection and rotary semantics.
pub struct Qwen3ModelSpec {
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
  priv weight_dtype : @spec.ModelDType
  priv kv_dtype : @spec.ModelDType
  priv tied_embeddings : Bool
} derive(Debug, Eq)

///|
/// Immutable dense Qwen3 metadata bound to independently verified content.
pub struct Qwen3ModelMetadata {
  priv spec : Qwen3ModelSpec
  priv content_digest : @spec.ContentDigest
} derive(Debug, Eq)

///|
/// Constructs the dense Qwen3 semantic profile.
pub fn Qwen3ModelSpec::new(
  vocabulary_size~ : Int,
  hidden_size~ : Int,
  intermediate_size~ : Int,
  layer_count~ : Int,
  attention_head_count~ : Int,
  kv_head_count~ : Int,
  head_dimension~ : Int,
  context_length~ : Int,
  rope_theta~ : Double,
  rms_norm_epsilon? : Double = 0.000001,
  weight_dtype~ : @spec.ModelDType,
  kv_dtype~ : @spec.ModelDType,
  tied_embeddings~ : Bool,
) -> Qwen3ModelSpec raise Qwen3ModelSpecError {
  let spec : Qwen3ModelSpec = {
    vocabulary_size,
    hidden_size,
    intermediate_size,
    layer_count,
    attention_head_count,
    kv_head_count,
    head_dimension,
    context_length,
    rope_theta,
    rms_norm_epsilon,
    weight_dtype,
    kv_dtype,
    tied_embeddings,
  }
  spec.validate()
  spec
}

///|
fn Qwen3ModelSpec::validate(self : Self) -> Unit raise Qwen3ModelSpecError {
  if self.vocabulary_size <= 0 {
    raise InvalidDimension(Vocabulary)
  }
  if self.hidden_size <= 0 {
    raise InvalidDimension(Hidden)
  }
  if self.intermediate_size <= 0 {
    raise InvalidDimension(Intermediate)
  }
  if self.layer_count <= 0 || self.layer_count > 454 {
    raise InvalidDimension(LayerCount)
  }
  if self.attention_head_count <= 0 {
    raise InvalidDimension(AttentionHeadCount)
  }
  if self.kv_head_count <= 0 {
    raise InvalidDimension(KvHeadCount)
  }
  if self.head_dimension <= 0 {
    raise InvalidDimension(HeadDimension)
  }
  if self.context_length <= 0 {
    raise InvalidDimension(ContextLength)
  }
  if self.kv_head_count > self.attention_head_count {
    raise InvalidHeadRelationship(KvHeadsExceedAttentionHeads)
  }
  if self.attention_head_count % self.kv_head_count != 0 {
    raise InvalidHeadRelationship(AttentionHeadsNotDivisibleByKvHeads)
  }
  if self.head_dimension % 2 != 0 {
    raise InvalidHeadRelationship(OddHeadDimension)
  }
  if self.attention_head_count > 2147483647 / self.head_dimension ||
    self.kv_head_count > 2147483647 / self.head_dimension {
    raise InvalidHeadRelationship(ProjectionWidthOverflow)
  }
  if self.rope_theta.is_nan() ||
    self.rope_theta.is_inf() ||
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
  if !self.tied_embeddings {
    raise UntiedEmbeddings
  }
}

///|
pub fn Qwen3ModelSpec::vocabulary_size(self : Self) -> Int {
  self.vocabulary_size
}

///|
pub fn Qwen3ModelSpec::hidden_size(self : Self) -> Int {
  self.hidden_size
}

///|
pub fn Qwen3ModelSpec::intermediate_size(self : Self) -> Int {
  self.intermediate_size
}

///|
pub fn Qwen3ModelSpec::layer_count(self : Self) -> Int {
  self.layer_count
}

///|
pub fn Qwen3ModelSpec::attention_head_count(self : Self) -> Int {
  self.attention_head_count
}

///|
/// Explicit query-head count. This name prevents consumers from mistaking
/// Qwen3's query projection width for the hidden width.
pub fn Qwen3ModelSpec::query_head_count(self : Self) -> Int {
  self.attention_head_count
}

///|
pub fn Qwen3ModelSpec::kv_head_count(self : Self) -> Int {
  self.kv_head_count
}

///|
pub fn Qwen3ModelSpec::head_dimension(self : Self) -> Int {
  self.head_dimension
}

///|
pub fn Qwen3ModelSpec::query_projection_width(self : Self) -> Int {
  self.attention_head_count * self.head_dimension
}

///|
pub fn Qwen3ModelSpec::kv_projection_width(self : Self) -> Int {
  self.kv_head_count * self.head_dimension
}

///|
pub fn Qwen3ModelSpec::context_length(self : Self) -> Int {
  self.context_length
}

///|
pub fn Qwen3ModelSpec::rope_theta(self : Self) -> Double {
  self.rope_theta
}

///|
pub fn Qwen3ModelSpec::rms_norm_epsilon(self : Self) -> Double {
  self.rms_norm_epsilon
}

///|
pub fn Qwen3ModelSpec::weight_dtype(self : Self) -> @spec.ModelDType {
  self.weight_dtype
}

///|
pub fn Qwen3ModelSpec::kv_dtype(self : Self) -> @spec.ModelDType {
  self.kv_dtype
}

///|
pub fn Qwen3ModelSpec::tied_embeddings(self : Self) -> Bool {
  self.tied_embeddings
}

///|
/// Dense Qwen3 applies RMSNorm independently to projected query and key heads
/// before rotary embedding. This is an unconditional architecture semantic.
pub fn Qwen3ModelSpec::has_query_key_norm(self : Self) -> Bool {
  ignore(self)
  true
}

///|
pub fn Qwen3ModelMetadata::from_verified_content(
  spec : Qwen3ModelSpec,
  content : @spec.ContentDigest,
) -> Qwen3ModelMetadata raise Qwen3ModelSpecError {
  spec.validate()
  // ContentDigest is opaque and its public constructors validate SHA-256
  // syntax. This package never converts that claim into plan authority.
  { spec, content_digest: content }
}

///|
pub fn Qwen3ModelMetadata::spec(self : Self) -> Qwen3ModelSpec {
  self.spec
}

///|
pub fn Qwen3ModelMetadata::content_digest(self : Self) -> @spec.ContentDigest {
  self.content_digest
}
