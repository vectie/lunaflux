///|
/// Bounded, payload-safe failures from the correctness-only decoder reference.
/// No error retains caller-provided floating-point values.
pub(all) suberror AdvancedDecoderReferenceError {
  InvalidLimit
  InvalidDimension
  InvalidTopK
  InvalidIterationCount
  NonFiniteInput(Int)
  NonFiniteParameter
  ZeroNormalization
  TokenOutOfRange
  InvalidHashTable
  ExpertIndexOutOfRange(Int)
  DuplicateExpertIndex(Int)
  InvalidCompression
  KeyIndexOutOfRange(Int)
  InvalidIndexLayout
  InvalidAuxiliaryOrder
  InvalidBlock
} derive(Debug, Eq)

///|
/// Explicit per-call ceilings for the allocation-using correctness oracle.
/// This package is never a production request-path fallback.
pub struct ReferenceLimits {
  priv max_values : Int
  priv max_width : Int
  priv max_iterations : Int
} derive(Debug, Eq)

///|
pub fn ReferenceLimits::new(
  max_values~ : Int,
  max_width~ : Int,
  max_iterations~ : Int,
) -> ReferenceLimits raise AdvancedDecoderReferenceError {
  if max_values <= 0 || max_width <= 0 || max_iterations <= 0 {
    raise InvalidLimit
  }
  { max_values, max_width, max_iterations, }
}

///|
pub fn ReferenceLimits::max_values(self : Self) -> Int {
  self.max_values
}

///|
pub fn ReferenceLimits::max_width(self : Self) -> Int {
  self.max_width
}

///|
pub fn ReferenceLimits::max_iterations(self : Self) -> Int {
  self.max_iterations
}

///|
pub struct ExpertRouting {
  priv expert_indices : ReadOnlyArray[Int]
  priv weights : ReadOnlyArray[Float]
} derive(Debug, Eq)

///|
pub fn ExpertRouting::expert_indices(self : Self) -> ReadOnlyArray[Int] {
  self.expert_indices
}

///|
pub fn ExpertRouting::weights(self : Self) -> ReadOnlyArray[Float] {
  self.weights
}

///|
pub fn ExpertRouting::selected_count(self : Self) -> Int {
  self.expert_indices.length()
}

///|
pub struct ReferenceTopK {
  priv indices : ReadOnlyArray[Int]
  priv scores : ReadOnlyArray[Float]
} derive(Debug, Eq)

///|
pub fn ReferenceTopK::indices(self : Self) -> ReadOnlyArray[Int] {
  self.indices
}

///|
/// F32 controls emitted by the versioned block-control reference. Arrays are
/// row-major `[row,stream]`, `[row,stream]`, and `[row,destination,source]`.
pub struct HyperConnectionBlockControls {
  priv pre : ReadOnlyArray[Float]
  priv post : ReadOnlyArray[Float]
  priv combination : ReadOnlyArray[Float]
} derive(Debug, Eq)

///|
pub fn HyperConnectionBlockControls::pre(self : Self) -> ReadOnlyArray[Float] {
  self.pre
}

///|
pub fn HyperConnectionBlockControls::post(self : Self) -> ReadOnlyArray[Float] {
  self.post
}

///|
pub fn HyperConnectionBlockControls::combination(
  self : Self,
) -> ReadOnlyArray[Float] {
  self.combination
}

///|
pub fn ReferenceTopK::scores(self : Self) -> ReadOnlyArray[Float] {
  self.scores
}

///|
pub(all) enum AuxiliaryReferenceStage {
  MtpInputProjection
  DsparkMainProjection
  DsparkNoiseEmbedding
  AuxiliaryDecoderStage(Int)
  AuxiliaryOutputHead
  DsparkMarkovAdjustment
  DsparkConfidence
} derive(Debug, Eq)

///|
/// Immutable inputs for the bounded end-to-end advanced decoder-block oracle.
/// All matrices are row-major. Expert matrices are expert-major, then row-major.
pub struct TinyAdvancedDecoderBlockInput {
  priv stream_activations : ReadOnlyArray[Float]
  priv query_weight : ReadOnlyArray[Float]
  priv keys : ReadOnlyArray[Float]
  priv values : ReadOnlyArray[Float]
  priv output_weight : ReadOnlyArray[Float]
  priv mixing_logits : ReadOnlyArray[Float]
  priv router_weight : ReadOnlyArray[Float]
  priv selection_bias : ReadOnlyArray[Float]
  priv expert_up_weight : ReadOnlyArray[Float]
  priv expert_gate_weight : ReadOnlyArray[Float]
  priv expert_down_weight : ReadOnlyArray[Float]
  priv stream_count : Int
  priv width : Int
  priv index_top_k : Int
  priv expert_count : Int
  priv experts_per_token : Int
  priv expert_hidden_width : Int
  priv sinkhorn_iterations : Int
  priv epsilon : Float
}

///|
pub struct TinyAdvancedDecoderBlockOutput {
  priv selected_key_indices : ReadOnlyArray[Int]
  priv attention_weights : ReadOnlyArray[Float]
  priv routing : ExpertRouting
  priv mixed_streams : ReadOnlyArray[Float]
  priv hidden : ReadOnlyArray[Float]
} derive(Debug, Eq)

///|
pub fn TinyAdvancedDecoderBlockOutput::selected_key_indices(
  self : Self,
) -> ReadOnlyArray[Int] {
  self.selected_key_indices
}

///|
pub fn TinyAdvancedDecoderBlockOutput::attention_weights(
  self : Self,
) -> ReadOnlyArray[Float] {
  self.attention_weights
}

///|
pub fn TinyAdvancedDecoderBlockOutput::routing(self : Self) -> ExpertRouting {
  self.routing
}

///|
pub fn TinyAdvancedDecoderBlockOutput::mixed_streams(
  self : Self,
) -> ReadOnlyArray[Float] {
  self.mixed_streams
}

///|
pub fn TinyAdvancedDecoderBlockOutput::hidden(
  self : Self,
) -> ReadOnlyArray[Float] {
  self.hidden
}

///|
fn finite(value : Float) -> Bool {
  !value.is_nan() && !value.is_inf()
}

///|
fn validate_values(
  values : ArrayView[Float],
  limits : ReferenceLimits,
) -> Unit raise AdvancedDecoderReferenceError {
  if values.is_empty() || values.length() > limits.max_values {
    raise InvalidDimension
  }
  for index, value in values {
    if !finite(value) {
      raise NonFiniteInput(index)
    }
  }
}

///|
fn validate_shape(
  count : Int,
  width : Int,
  limits : ReferenceLimits,
) -> Unit raise AdvancedDecoderReferenceError {
  if count <= 0 ||
    width <= 0 ||
    width > limits.max_width ||
    count > limits.max_values / width {
    raise InvalidDimension
  }
}

///|
fn combined_values_within_limit(
  first : Int,
  second : Int,
  third : Int,
  limits : ReferenceLimits,
) -> Bool {
  Int64::from_int(first) + Int64::from_int(second) + Int64::from_int(third) <=
  Int64::from_int(limits.max_values)
}
