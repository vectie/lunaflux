///|
/// Decoder block selected by the exact Qwen3.5-MoE hybrid schedule.
pub(all) enum Qwen35HybridLayerKind {
  GatedDeltaLinearAttention
  FullAttention
} derive(Debug, Eq)

///|
/// Quantized matrix storage admitted by this Qwen-only profile.
pub(all) enum Qwen35MatrixDType {
  Fp8E4M3
} derive(Debug, Eq)

///|
/// Scale and non-quantized execution storage used by this profile.
pub(all) enum Qwen35AuxiliaryDType {
  BFloat16
  Float32
} derive(Debug, Eq)

///|
/// Exact, text-only semantic profile for the pinned
/// Qwen3.6-35B-A3B-FP8 artifact.
///
/// This type is deliberately separate from dense Qwen3. It grants no vision
/// or multi-token-prediction execution authority, and it does not imply that
/// only the active experts need to be resident in device memory.
pub struct Qwen35MoeTextSpec {
  priv profile_version : Int
} derive(Debug, Eq)

///|
/// Immutable parsed config metadata bound to independently verified content.
pub struct Qwen35MoeTextMetadata {
  priv spec : Qwen35MoeTextSpec
  priv content_digest : @spec.ContentDigest
} derive(Debug, Eq)

///|
/// Constructs the one exact Qwen3.5-MoE text profile currently understood by
/// LunaFlux. Numeric tensor and kernel admission remain separate boundaries.
pub fn Qwen35MoeTextSpec::pinned_35b_a3b_fp8_text_only() -> Self {
  { profile_version: 1 }
}

///|
pub fn Qwen35MoeTextSpec::vocabulary_size(self : Self) -> Int {
  ignore(self)
  248320
}

///|
pub fn Qwen35MoeTextSpec::hidden_size(self : Self) -> Int {
  ignore(self)
  2048
}

///|
pub fn Qwen35MoeTextSpec::layer_count(self : Self) -> Int {
  ignore(self)
  40
}

///|
pub fn Qwen35MoeTextSpec::attention_head_count(self : Self) -> Int {
  ignore(self)
  16
}

///|
pub fn Qwen35MoeTextSpec::kv_head_count(self : Self) -> Int {
  ignore(self)
  2
}

///|
pub fn Qwen35MoeTextSpec::head_dimension(self : Self) -> Int {
  ignore(self)
  256
}

///|
pub fn Qwen35MoeTextSpec::context_length(self : Self) -> Int {
  ignore(self)
  262144
}

///|
/// Returns the exact three-linear/one-full repeating block schedule.
pub fn Qwen35MoeTextSpec::layer_kind(
  self : Self,
  layer_index : Int,
) -> Qwen35HybridLayerKind? {
  if layer_index < 0 || layer_index >= self.layer_count() {
    None
  } else if layer_index % 4 == 3 {
    Some(FullAttention)
  } else {
    Some(GatedDeltaLinearAttention)
  }
}

///|
pub fn Qwen35MoeTextSpec::full_attention_interval(self : Self) -> Int {
  ignore(self)
  4
}

///|
pub fn Qwen35MoeTextSpec::attention_output_gate(self : Self) -> Bool {
  ignore(self)
  true
}

///|
pub fn Qwen35MoeTextSpec::linear_key_head_count(self : Self) -> Int {
  ignore(self)
  16
}

///|
pub fn Qwen35MoeTextSpec::linear_key_head_dimension(self : Self) -> Int {
  ignore(self)
  128
}

///|
pub fn Qwen35MoeTextSpec::linear_value_head_count(self : Self) -> Int {
  ignore(self)
  32
}

///|
pub fn Qwen35MoeTextSpec::linear_value_head_dimension(self : Self) -> Int {
  ignore(self)
  128
}

///|
pub fn Qwen35MoeTextSpec::linear_convolution_width(self : Self) -> Int {
  ignore(self)
  4
}

///|
pub fn Qwen35MoeTextSpec::recurrent_state_dtype(
  self : Self,
) -> Qwen35AuxiliaryDType {
  ignore(self)
  Float32
}

///|
pub fn Qwen35MoeTextSpec::routed_expert_count(self : Self) -> Int {
  ignore(self)
  256
}

///|
pub fn Qwen35MoeTextSpec::experts_per_token(self : Self) -> Int {
  ignore(self)
  8
}

///|
pub fn Qwen35MoeTextSpec::expert_intermediate_size(self : Self) -> Int {
  ignore(self)
  512
}

///|
pub fn Qwen35MoeTextSpec::shared_expert_intermediate_size(self : Self) -> Int {
  ignore(self)
  512
}

///|
pub fn Qwen35MoeTextSpec::has_learned_shared_expert_gate(self : Self) -> Bool {
  ignore(self)
  true
}

///|
pub fn Qwen35MoeTextSpec::rope_theta(self : Self) -> Double {
  ignore(self)
  10000000.0
}

///|
pub fn Qwen35MoeTextSpec::partial_rotary_factor(self : Self) -> Double {
  ignore(self)
  0.25
}

///|
pub fn Qwen35MoeTextSpec::mrope_section_count(self : Self) -> Int {
  ignore(self)
  3
}

///|
pub fn Qwen35MoeTextSpec::mrope_section(self : Self, index : Int) -> Int? {
  ignore(self)
  match index {
    0 | 1 => Some(11)
    2 => Some(10)
    _ => None
  }
}

///|
pub fn Qwen35MoeTextSpec::mrope_interleaved(self : Self) -> Bool {
  ignore(self)
  true
}

///|
pub fn Qwen35MoeTextSpec::matrix_dtype(self : Self) -> Qwen35MatrixDType {
  ignore(self)
  Fp8E4M3
}

///|
pub fn Qwen35MoeTextSpec::inverse_scale_dtype(
  self : Self,
) -> Qwen35AuxiliaryDType {
  ignore(self)
  BFloat16
}

///|
pub fn Qwen35MoeTextSpec::fp8_block_rows(self : Self) -> Int {
  ignore(self)
  128
}

///|
pub fn Qwen35MoeTextSpec::fp8_block_columns(self : Self) -> Int {
  ignore(self)
  128
}

///|
pub fn Qwen35MoeTextSpec::uses_dynamic_activation_scaling(self : Self) -> Bool {
  ignore(self)
  true
}

///|
pub fn Qwen35MoeTextSpec::tied_embeddings(self : Self) -> Bool {
  ignore(self)
  false
}

///|
/// The source config declares one MTP layer, but the first executable route
/// excludes it explicitly.
pub fn Qwen35MoeTextSpec::source_mtp_layer_count(self : Self) -> Int {
  ignore(self)
  1
}

///|
pub fn Qwen35MoeTextSpec::mtp_execution_enabled(self : Self) -> Bool {
  ignore(self)
  false
}

///|
/// The selected artifact has a vision sub-config, but this profile owns only
/// text execution.
pub fn Qwen35MoeTextSpec::vision_execution_enabled(self : Self) -> Bool {
  ignore(self)
  false
}

///|
pub fn Qwen35MoeTextMetadata::from_verified_config(
  content_digest : @spec.ContentDigest,
) -> Self {
  { spec: Qwen35MoeTextSpec::pinned_35b_a3b_fp8_text_only(), content_digest }
}

///|
pub fn Qwen35MoeTextMetadata::spec(self : Self) -> Qwen35MoeTextSpec {
  self.spec
}

///|
pub fn Qwen35MoeTextMetadata::content_digest(
  self : Self,
) -> @spec.ContentDigest {
  self.content_digest
}
