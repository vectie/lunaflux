///|
/// Exact official DeepSeek-V4 checkpoint profile admitted by LunaFlux schema
/// v1. There is intentionally no arbitrary V4-like configuration variant.
pub(all) enum DeepSeekV4Profile {
  Flash
  FlashBase
  Pro
  ProBase
  Flash0731
} derive(Debug, Eq)

///|
/// Stored expert representation declared by an exact official checkpoint.
pub(all) enum DeepSeekV4ExpertStorage {
  Fp4E2M1
  Fp8E4M3
} derive(Debug, Eq)

///|
/// Fail-closed profile selection errors. Model identifiers are payload-safe
/// and are never included in the error.
pub(all) suberror DeepSeekV4SpecError {
  UnsupportedProfile
} derive(Debug, Eq)

///|
/// Validated, closed DeepSeek-V4 architecture semantics.
///
/// All dimensions are projected from `profile`; callers cannot construct a
/// configuration that merely resembles DeepSeek-V4. Adding another checkpoint
/// therefore requires an explicit schema change and official source evidence.
pub struct DeepSeekV4ModelSpec {
  priv profile : DeepSeekV4Profile
} derive(Debug, Eq)

///|
/// Immutable DeepSeek-V4 metadata bound to independently verified content.
pub struct DeepSeekV4ModelMetadata {
  priv spec : DeepSeekV4ModelSpec
  priv content_digest : @spec.ContentDigest
} derive(Debug, Eq)

///|
/// Admits only exact official Hugging Face repository identifiers covered by
/// schema v1. Aliases, case variants, revisions, and unknown future releases
/// fail closed.
pub fn DeepSeekV4ModelSpec::from_official_model_id(
  model_id : StringView,
) -> DeepSeekV4ModelSpec raise DeepSeekV4SpecError {
  let profile = match model_id {
    "deepseek-ai/DeepSeek-V4-Flash" => Flash
    "deepseek-ai/DeepSeek-V4-Flash-Base" => FlashBase
    "deepseek-ai/DeepSeek-V4-Pro" => Pro
    "deepseek-ai/DeepSeek-V4-Pro-Base" => ProBase
    "deepseek-ai/DeepSeek-V4-Flash-0731" => Flash0731
    _ => raise UnsupportedProfile
  }
  { profile, }
}

///|
pub fn DeepSeekV4ModelSpec::profile(self : Self) -> DeepSeekV4Profile {
  self.profile
}

///|
pub fn DeepSeekV4ModelSpec::architecture(self : Self) -> String {
  ignore(self)
  "DeepseekV4ForCausalLM"
}

///|
pub fn DeepSeekV4ModelSpec::model_type(self : Self) -> String {
  ignore(self)
  "deepseek_v4"
}

///|
/// Stable profile identity used by the advanced semantic execution-plan
/// canonical encoder. The schema suffix prevents a future profile expansion
/// from aliasing an existing plan.
pub fn DeepSeekV4ModelSpec::execution_profile_identity(self : Self) -> String {
  match self.profile {
    Flash => "deepseek-ai/DeepSeek-V4-Flash/v1"
    FlashBase => "deepseek-ai/DeepSeek-V4-Flash-Base/v1"
    Pro => "deepseek-ai/DeepSeek-V4-Pro/v1"
    ProBase => "deepseek-ai/DeepSeek-V4-Pro-Base/v1"
    Flash0731 => "deepseek-ai/DeepSeek-V4-Flash-0731/v1"
  }
}

///|
pub fn DeepSeekV4ModelSpec::vocabulary_size(self : Self) -> Int {
  ignore(self)
  129280
}

///|
pub fn DeepSeekV4ModelSpec::hidden_size(self : Self) -> Int {
  match self.profile {
    Flash | FlashBase | Flash0731 => 4096
    Pro | ProBase => 7168
  }
}

///|
pub fn DeepSeekV4ModelSpec::moe_intermediate_size(self : Self) -> Int {
  match self.profile {
    Flash | FlashBase | Flash0731 => 2048
    Pro | ProBase => 3072
  }
}

///|
pub fn DeepSeekV4ModelSpec::layer_count(self : Self) -> Int {
  match self.profile {
    Flash | FlashBase | Flash0731 => 43
    Pro | ProBase => 61
  }
}

///|
pub fn DeepSeekV4ModelSpec::attention_head_count(self : Self) -> Int {
  match self.profile {
    Flash | FlashBase | Flash0731 => 64
    Pro | ProBase => 128
  }
}

///|
pub fn DeepSeekV4ModelSpec::kv_head_count(self : Self) -> Int {
  ignore(self)
  1
}

///|
pub fn DeepSeekV4ModelSpec::head_dimension(self : Self) -> Int {
  ignore(self)
  512
}

///|
pub fn DeepSeekV4ModelSpec::rope_head_dimension(self : Self) -> Int {
  ignore(self)
  64
}

///|
pub fn DeepSeekV4ModelSpec::q_lora_rank(self : Self) -> Int {
  match self.profile {
    Flash | FlashBase | Flash0731 => 1024
    Pro | ProBase => 1536
  }
}

///|
pub fn DeepSeekV4ModelSpec::o_lora_rank(self : Self) -> Int {
  ignore(self)
  1024
}

///|
pub fn DeepSeekV4ModelSpec::output_group_count(self : Self) -> Int {
  match self.profile {
    Flash | FlashBase | Flash0731 => 8
    Pro | ProBase => 16
  }
}

///|
pub fn DeepSeekV4ModelSpec::routed_expert_count(self : Self) -> Int {
  match self.profile {
    Flash | FlashBase | Flash0731 => 256
    Pro | ProBase => 384
  }
}

///|
pub fn DeepSeekV4ModelSpec::shared_expert_count(self : Self) -> Int {
  ignore(self)
  1
}

///|
pub fn DeepSeekV4ModelSpec::experts_per_token(self : Self) -> Int {
  ignore(self)
  6
}

///|
pub fn DeepSeekV4ModelSpec::routed_scaling_factor(self : Self) -> Double {
  match self.profile {
    Flash | FlashBase | Flash0731 => 1.5
    Pro | ProBase => 2.5
  }
}

///|
pub fn DeepSeekV4ModelSpec::expert_storage(
  self : Self,
) -> DeepSeekV4ExpertStorage {
  match self.profile {
    Flash | Pro | Flash0731 => Fp4E2M1
    FlashBase | ProBase => Fp8E4M3
  }
}

///|
pub fn DeepSeekV4ModelSpec::max_position_embeddings(self : Self) -> Int {
  ignore(self)
  1048576
}

///|
pub fn DeepSeekV4ModelSpec::sliding_window(self : Self) -> Int {
  ignore(self)
  128
}

///|
pub fn DeepSeekV4ModelSpec::rope_theta(self : Self) -> Double {
  ignore(self)
  10000.0
}

///|
pub fn DeepSeekV4ModelSpec::compressed_rope_theta(self : Self) -> Double {
  ignore(self)
  160000.0
}

///|
pub fn DeepSeekV4ModelSpec::original_position_embeddings(self : Self) -> Int {
  ignore(self)
  65536
}

///|
pub fn DeepSeekV4ModelSpec::rope_scaling_factor(self : Self) -> Double {
  ignore(self)
  16.0
}

///|
pub fn DeepSeekV4ModelSpec::rope_beta_fast(self : Self) -> Double {
  ignore(self)
  32.0
}

///|
pub fn DeepSeekV4ModelSpec::rope_beta_slow(self : Self) -> Double {
  ignore(self)
  1.0
}

///|
pub fn DeepSeekV4ModelSpec::rms_norm_epsilon(self : Self) -> Double {
  ignore(self)
  0.000001
}

///|
pub fn DeepSeekV4ModelSpec::hyper_connection_width(self : Self) -> Int {
  ignore(self)
  4
}

///|
pub fn DeepSeekV4ModelSpec::hyper_connection_epsilon(self : Self) -> Double {
  ignore(self)
  0.000001
}

///|
pub fn DeepSeekV4ModelSpec::hyper_connection_sinkhorn_iterations(
  self : Self,
) -> Int {
  ignore(self)
  20
}

///|
pub fn DeepSeekV4ModelSpec::index_head_count(self : Self) -> Int {
  ignore(self)
  64
}

///|
pub fn DeepSeekV4ModelSpec::index_head_dimension(self : Self) -> Int {
  ignore(self)
  128
}

///|
pub fn DeepSeekV4ModelSpec::index_top_k(self : Self) -> Int {
  match self.profile {
    Flash | FlashBase | Flash0731 => 512
    Pro | ProBase => 1024
  }
}

///|
pub fn DeepSeekV4ModelSpec::normalizes_selected_expert_probabilities(
  self : Self,
) -> Bool {
  ignore(self)
  true
}

///|
pub fn DeepSeekV4ModelSpec::swiglu_limit(self : Self) -> Double {
  ignore(self)
  10.0
}

///|
pub fn DeepSeekV4ModelSpec::hash_layer_count(self : Self) -> Int {
  ignore(self)
  3
}

///|
pub fn DeepSeekV4ModelSpec::next_token_prediction_layer_count(
  self : Self,
) -> Int {
  ignore(self)
  1
}

///|
/// Number of prediction-stage blocks physically present in the official
/// checkpoint. Flash-0731 retains the upstream configuration value `1` for
/// `num_nextn_predict_layers`, but its DSpark artifact contains `mtp.0` through
/// `mtp.2`; execution and tensor binding must therefore use this artifact
/// count rather than reinterpret the configuration field.
pub fn DeepSeekV4ModelSpec::prediction_stage_count(self : Self) -> Int {
  match self.profile {
    Flash0731 => 3
    Flash | FlashBase | Pro | ProBase => 1
  }
}

///|
pub fn DeepSeekV4ModelSpec::has_dspark(self : Self) -> Bool {
  self.profile is Flash0731
}

///|
pub fn DeepSeekV4ModelSpec::dspark_block_size(self : Self) -> Int {
  if self.has_dspark() {
    5
  } else {
    0
  }
}

///|
pub fn DeepSeekV4ModelSpec::dspark_noise_token_id(self : Self) -> Int {
  if self.has_dspark() {
    128799
  } else {
    0
  }
}

///|
pub fn DeepSeekV4ModelSpec::dspark_target_layer_ids(
  self : Self,
) -> ReadOnlyArray[Int] {
  if self.has_dspark() {
    [40, 41, 42]
  } else {
    []
  }
}

///|
pub fn DeepSeekV4ModelSpec::dspark_markov_rank(self : Self) -> Int {
  if self.has_dspark() {
    256
  } else {
    0
  }
}

///|
pub fn DeepSeekV4ModelSpec::attention_profile_count(self : Self) -> Int {
  match self.profile {
    Flash | FlashBase => 44
    Pro | ProBase => 62
    Flash0731 => 46
  }
}

///|
/// Returns the exact official `compress_ratios` entry. The list includes the
/// base decoder layers and official attached MTP/DSpark modules. An invalid
/// index returns `None`; it never aliases a nearby layer.
pub fn DeepSeekV4ModelSpec::compression_ratio(self : Self, index : Int) -> Int? {
  if index < 0 || index >= self.attention_profile_count() {
    return None
  }
  match self.profile {
    Flash | FlashBase | Flash0731 =>
      if index < 2 || index >= 43 {
        Some(0)
      } else if index % 2 == 0 {
        Some(4)
      } else {
        Some(128)
      }
    Pro | ProBase =>
      if index < 2 {
        Some(128)
      } else if index >= 61 {
        Some(0)
      } else if index % 2 == 0 {
        Some(4)
      } else {
        Some(128)
      }
  }
}

///|
pub fn DeepSeekV4ModelMetadata::from_verified_content(
  spec : DeepSeekV4ModelSpec,
  content : @spec.ContentDigest,
) -> DeepSeekV4ModelMetadata {
  { spec, content_digest: content, }
}

///|
pub fn DeepSeekV4ModelMetadata::spec(self : Self) -> DeepSeekV4ModelSpec {
  self.spec
}

///|
pub fn DeepSeekV4ModelMetadata::content_digest(
  self : Self,
) -> @spec.ContentDigest {
  self.content_digest
}
