///|
/// Validated dimensions for the bounded full-attention-compatible Mistral
/// decoder profile.
///
/// Mistral's sliding attention is equivalent to ordinary causal attention
/// while a request stays within one window. `context_length` is therefore the
/// smaller of the declared position ceiling and an optional sliding window.
/// The semantic plan never needs a scheduler or KV-family branch.
pub struct MistralModelSpec {
  priv vocabulary_size : Int32
  priv hidden_size : Int32
  priv intermediate_size : Int32
  priv layer_count : Int32
  priv attention_head_count : Int32
  priv kv_head_count : Int32
  priv head_dimension : Int32
  priv max_position_embeddings : Int32
  priv sliding_window : Int32?
  priv context_length : Int32
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
  vocabulary_size~ : Int32,
  hidden_size~ : Int32,
  intermediate_size~ : Int32,
  layer_count~ : Int32,
  attention_head_count~ : Int32,
  kv_head_count~ : Int32,
  max_position_embeddings~ : Int32,
  sliding_window? : Int32,
  rope_theta~ : Double,
  rms_norm_epsilon~ : Double = 0.000001,
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
pub fn MistralModelSpec::vocabulary_size(self : Self) -> Int32 {
  self.vocabulary_size
}

///|
pub fn MistralModelSpec::hidden_size(self : Self) -> Int32 {
  self.hidden_size
}

///|
pub fn MistralModelSpec::intermediate_size(self : Self) -> Int32 {
  self.intermediate_size
}

///|
pub fn MistralModelSpec::layer_count(self : Self) -> Int32 {
  self.layer_count
}

///|
pub fn MistralModelSpec::attention_head_count(self : Self) -> Int32 {
  self.attention_head_count
}

///|
pub fn MistralModelSpec::kv_head_count(self : Self) -> Int32 {
  self.kv_head_count
}

///|
pub fn MistralModelSpec::head_dimension(self : Self) -> Int32 {
  self.head_dimension
}

///|
pub fn MistralModelSpec::max_position_embeddings(self : Self) -> Int32 {
  self.max_position_embeddings
}

///|
pub fn MistralModelSpec::sliding_window(self : Self) -> Int32? {
  self.sliding_window
}

///|
pub fn MistralModelSpec::context_length(self : Self) -> Int32 {
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
  { spec, content_digest: content, }
}

///|
pub fn MistralModelMetadata::spec(self : Self) -> MistralModelSpec {
  self.spec
}

///|
pub fn MistralModelMetadata::content_digest(self : Self) -> @spec.ContentDigest {
  self.content_digest
}

///|
pub extend MistralModelSpec with @debug.Debug::{to_repr}

///|
pub extend MistralModelSpec with Eq::{not_equal, equal}

///|
pub extend MistralModelMetadata with @debug.Debug::{to_repr}

///|
pub extend MistralModelMetadata with Eq::{not_equal, equal}
