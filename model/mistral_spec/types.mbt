///|
/// Validated dimensions for the bounded full-attention-compatible Mistral
/// decoder profile.
///
/// Mistral's sliding attention is equivalent to ordinary causal attention
/// while a request stays within one window. `context_length` is therefore the
/// smaller of the declared position ceiling and an optional sliding window.
/// The semantic plan never needs a scheduler or KV-family branch.
pub struct MistralModelSpec {
  priv vocabulary_size : Int
  priv hidden_size : Int
  priv intermediate_size : Int
  priv layer_count : Int
  priv attention_head_count : Int
  priv kv_head_count : Int
  priv head_dimension : Int
  priv max_position_embeddings : Int
  priv sliding_window : Int?
  priv context_length : Int
  priv rope_theta : Double
  priv rms_norm_epsilon : Double
  priv weight_dtype : @spec.ModelDType
  priv kv_dtype : @spec.ModelDType
} derive(Debug, Eq)

///|
/// Immutable Mistral metadata bound to independently verified model content.
pub struct MistralModelMetadata {
  priv spec : MistralModelSpec
  priv content_digest : @spec.ContentDigest
} derive(Debug, Eq)

///|
/// Validates the bounded Mistral v1 semantic profile.
///
/// The shared dense dimensions are checked by the established dense-Llama
/// semantic contract because the admitted operation graph is identical.
/// An optional sliding window only narrows the public context ceiling; LunaFlux
/// never silently runs a sequence whose attention semantics would differ.
pub fn MistralModelSpec::new(
  vocabulary_size~ : Int,
  hidden_size~ : Int,
  intermediate_size~ : Int,
  layer_count~ : Int,
  attention_head_count~ : Int,
  kv_head_count~ : Int,
  max_position_embeddings~ : Int,
  sliding_window? : Int,
  rope_theta~ : Double,
  rms_norm_epsilon? : Double = 0.000001,
  weight_dtype~ : @spec.ModelDType,
  kv_dtype~ : @spec.ModelDType,
) -> MistralModelSpec raise @spec.ModelSpecError {
  if max_position_embeddings <= 0 {
    raise InvalidDimension(ContextLength)
  }
  let context_length = match sliding_window {
    Some(window) => {
      if window <= 0 {
        raise InvalidDimension(ContextLength)
      }
      window.min(max_position_embeddings)
    }
    None => max_position_embeddings
  }
  let dense = @spec.LlamaModelSpec::new(
    vocabulary_size~,
    hidden_size~,
    intermediate_size~,
    layer_count~,
    attention_head_count~,
    kv_head_count~,
    context_length~,
    rope_theta~,
    rms_norm_epsilon~,
    weight_dtype~,
    kv_dtype~,
  )
  {
    vocabulary_size: dense.vocabulary_size(),
    hidden_size: dense.hidden_size(),
    intermediate_size: dense.intermediate_size(),
    layer_count: dense.layer_count(),
    attention_head_count: dense.attention_head_count(),
    kv_head_count: dense.kv_head_count(),
    head_dimension: dense.head_dimension(),
    max_position_embeddings,
    sliding_window,
    context_length: dense.context_length(),
    rope_theta: dense.rope_theta(),
    rms_norm_epsilon: dense.rms_norm_epsilon(),
    weight_dtype: dense.weight_dtype(),
    kv_dtype: dense.kv_dtype(),
  }
}

///|
pub fn MistralModelSpec::vocabulary_size(self : Self) -> Int {
  self.vocabulary_size
}

///|
pub fn MistralModelSpec::hidden_size(self : Self) -> Int {
  self.hidden_size
}

///|
pub fn MistralModelSpec::intermediate_size(self : Self) -> Int {
  self.intermediate_size
}

///|
pub fn MistralModelSpec::layer_count(self : Self) -> Int {
  self.layer_count
}

///|
pub fn MistralModelSpec::attention_head_count(self : Self) -> Int {
  self.attention_head_count
}

///|
pub fn MistralModelSpec::kv_head_count(self : Self) -> Int {
  self.kv_head_count
}

///|
pub fn MistralModelSpec::head_dimension(self : Self) -> Int {
  self.head_dimension
}

///|
pub fn MistralModelSpec::max_position_embeddings(self : Self) -> Int {
  self.max_position_embeddings
}

///|
pub fn MistralModelSpec::sliding_window(self : Self) -> Int? {
  self.sliding_window
}

///|
pub fn MistralModelSpec::context_length(self : Self) -> Int {
  self.context_length
}

///|
pub fn MistralModelSpec::rope_theta(self : Self) -> Double {
  self.rope_theta
}

///|
pub fn MistralModelSpec::rms_norm_epsilon(self : Self) -> Double {
  self.rms_norm_epsilon
}

///|
pub fn MistralModelSpec::weight_dtype(self : Self) -> @spec.ModelDType {
  self.weight_dtype
}

///|
pub fn MistralModelSpec::kv_dtype(self : Self) -> @spec.ModelDType {
  self.kv_dtype
}

///|
pub fn MistralModelMetadata::from_verified_content(
  spec : MistralModelSpec,
  content : @spec.ContentDigest,
) -> MistralModelMetadata raise @spec.ModelSpecError {
  // Reconstructing the shared dense contract revalidates every field without
  // granting this metadata any execution-plan identity.
  ignore(
    @spec.LlamaModelSpec::new(
      vocabulary_size=spec.vocabulary_size(),
      hidden_size=spec.hidden_size(),
      intermediate_size=spec.intermediate_size(),
      layer_count=spec.layer_count(),
      attention_head_count=spec.attention_head_count(),
      kv_head_count=spec.kv_head_count(),
      context_length=spec.context_length(),
      rope_theta=spec.rope_theta(),
      rms_norm_epsilon=spec.rms_norm_epsilon(),
      weight_dtype=spec.weight_dtype(),
      kv_dtype=spec.kv_dtype(),
    ),
  )
  // `ContentDigest` is opaque and its only public constructor validates it.
  { spec, content_digest: content }
}

///|
pub fn MistralModelMetadata::spec(self : Self) -> MistralModelSpec {
  self.spec
}

///|
pub fn MistralModelMetadata::content_digest(self : Self) -> @spec.ContentDigest {
  self.content_digest
}
