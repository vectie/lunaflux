///|
pub fn JointDiffusionStage::new(
  kind~ : JointDiffusionStageKind,
  presence~ : JointDiffusionStagePresence,
) -> JointDiffusionStage {
  { kind, presence, }
}

///|
pub fn JointDiffusionStage::kind(self : Self) -> JointDiffusionStageKind {
  self.kind
}

///|
pub fn JointDiffusionStage::presence(
  self : Self,
) -> JointDiffusionStagePresence {
  self.presence
}

///|
pub fn JointDiffusionScheduler::new(
  latent~ : JointDiffusionLatentKind,
  kind~ : JointDiffusionSchedulerKind,
  shift_numerator~ : Int,
  shift_denominator~ : Int,
  training_timestep_count? : Int = 1000,
) -> JointDiffusionScheduler raise JointDiffusionPlanError {
  if shift_numerator <= 0 ||
    shift_denominator <= 0 ||
    training_timestep_count <= 0 ||
    training_timestep_count > 1000000 {
    raise InvalidScheduler
  }
  { latent, kind, shift_numerator, shift_denominator, training_timestep_count, }
}

///|
pub fn JointDiffusionScheduler::latent(self : Self) -> JointDiffusionLatentKind {
  self.latent
}

///|
pub fn JointDiffusionScheduler::kind(
  self : Self,
) -> JointDiffusionSchedulerKind {
  self.kind
}

///|
pub fn JointDiffusionScheduler::shift_numerator(self : Self) -> Int {
  self.shift_numerator
}

///|
pub fn JointDiffusionScheduler::shift_denominator(self : Self) -> Int {
  self.shift_denominator
}

///|
pub fn JointDiffusionScheduler::shift(self : Self) -> Double {
  self.shift_numerator.to_double() / self.shift_denominator.to_double()
}

///|
pub fn JointDiffusionScheduler::training_timestep_count(self : Self) -> Int {
  self.training_timestep_count
}

///|
pub fn JointDiffusionConditioningLimits::new(
  max_keyframes~ : Int,
  max_image_references~ : Int,
  max_video_references~ : Int,
  max_audio_references~ : Int,
  max_total_references~ : Int,
  allow_audio_only_references~ : Bool,
) -> JointDiffusionConditioningLimits raise JointDiffusionPlanError {
  if max_keyframes < 0 ||
    max_image_references < 0 ||
    max_video_references < 0 ||
    max_audio_references < 0 ||
    max_total_references < 0 ||
    max_image_references > max_total_references ||
    max_video_references > max_total_references ||
    max_audio_references > max_total_references {
    raise InvalidConditioningLimits
  }
  {
    max_keyframes,
    max_image_references,
    max_video_references,
    max_audio_references,
    max_total_references,
    allow_audio_only_references,
  }
}

///|
pub fn JointDiffusionConditioningLimits::max_keyframes(self : Self) -> Int {
  self.max_keyframes
}

///|
pub fn JointDiffusionConditioningLimits::max_image_references(
  self : Self,
) -> Int {
  self.max_image_references
}

///|
pub fn JointDiffusionConditioningLimits::max_video_references(
  self : Self,
) -> Int {
  self.max_video_references
}

///|
pub fn JointDiffusionConditioningLimits::max_audio_references(
  self : Self,
) -> Int {
  self.max_audio_references
}

///|
pub fn JointDiffusionConditioningLimits::max_total_references(
  self : Self,
) -> Int {
  self.max_total_references
}

///|
pub fn JointDiffusionConditioningLimits::allow_audio_only_references(
  self : Self,
) -> Bool {
  self.allow_audio_only_references
}

///|
pub fn JointDiffusionDenoiserShape::new(
  hidden_size~ : Int,
  layer_count~ : Int,
  refiner_layer_count~ : Int,
  attention_head_count~ : Int,
  attention_head_dimension~ : Int,
  attention_rotary_dimension? : Int = 2,
  query_key_norm_epsilon_numerator? : Int = 1,
  query_key_norm_epsilon_denominator? : Int = 100000,
  feed_forward_dimension~ : Int,
  text_conditioning_dimension~ : Int,
  timestep_frequency_dimension~ : Int,
  timestep_embedding_hidden_dimension? : Int = 4,
  final_adaln_conditioning_dimension? : Int = 2,
  final_norm_epsilon_numerator? : Int = 1,
  final_norm_epsilon_denominator? : Int = 100000,
) -> JointDiffusionDenoiserShape raise JointDiffusionPlanError {
  if hidden_size <= 0 ||
    layer_count <= 0 ||
    refiner_layer_count < 0 ||
    attention_head_count <= 0 ||
    attention_head_dimension <= 0 ||
    attention_rotary_dimension <= 0 ||
    attention_rotary_dimension > attention_head_dimension ||
    attention_rotary_dimension % 2 != 0 ||
    query_key_norm_epsilon_numerator <= 0 ||
    query_key_norm_epsilon_denominator <= 0 ||
    feed_forward_dimension <= 0 ||
    text_conditioning_dimension <= 0 ||
    timestep_frequency_dimension <= 0 ||
    timestep_embedding_hidden_dimension <= 0 ||
    final_adaln_conditioning_dimension <= 0 ||
    final_norm_epsilon_numerator <= 0 ||
    final_norm_epsilon_denominator <= 0 ||
    hidden_size > 1000000 ||
    layer_count > 1000000 ||
    refiner_layer_count > layer_count ||
    attention_head_count > 1000000 ||
    attention_head_dimension > 1000000 ||
    query_key_norm_epsilon_numerator > 1000000 ||
    query_key_norm_epsilon_denominator > 1000000000 ||
    feed_forward_dimension > 1000000 ||
    text_conditioning_dimension > 1000000 ||
    timestep_frequency_dimension > 1000000 ||
    timestep_embedding_hidden_dimension > 1000000 ||
    final_adaln_conditioning_dimension > 1000000 ||
    final_norm_epsilon_numerator > 1000000 ||
    final_norm_epsilon_denominator > 1000000000 {
    raise InvalidShapeContract
  }
  {
    hidden_size,
    layer_count,
    refiner_layer_count,
    attention_head_count,
    attention_head_dimension,
    attention_rotary_dimension,
    query_key_norm_epsilon_numerator,
    query_key_norm_epsilon_denominator,
    feed_forward_dimension,
    text_conditioning_dimension,
    timestep_frequency_dimension,
    timestep_embedding_hidden_dimension,
    final_adaln_conditioning_dimension,
    final_norm_epsilon_numerator,
    final_norm_epsilon_denominator,
  }
}

///|
pub fn JointDiffusionDenoiserShape::hidden_size(self : Self) -> Int {
  self.hidden_size
}

///|
pub fn JointDiffusionDenoiserShape::layer_count(self : Self) -> Int {
  self.layer_count
}

///|
pub fn JointDiffusionDenoiserShape::refiner_layer_count(self : Self) -> Int {
  self.refiner_layer_count
}

///|
pub fn JointDiffusionDenoiserShape::attention_head_count(self : Self) -> Int {
  self.attention_head_count
}

///|
pub fn JointDiffusionDenoiserShape::attention_head_dimension(
  self : Self,
) -> Int {
  self.attention_head_dimension
}

///|
pub fn JointDiffusionDenoiserShape::attention_width(self : Self) -> Int {
  self.attention_head_count * self.attention_head_dimension
}

///|
pub fn JointDiffusionDenoiserShape::attention_rotary_dimension(
  self : Self,
) -> Int {
  self.attention_rotary_dimension
}

///|
pub fn JointDiffusionDenoiserShape::query_key_norm_epsilon_numerator(
  self : Self,
) -> Int {
  self.query_key_norm_epsilon_numerator
}

///|
pub fn JointDiffusionDenoiserShape::query_key_norm_epsilon_denominator(
  self : Self,
) -> Int {
  self.query_key_norm_epsilon_denominator
}

///|
pub fn JointDiffusionDenoiserShape::feed_forward_dimension(self : Self) -> Int {
  self.feed_forward_dimension
}

///|
pub fn JointDiffusionDenoiserShape::text_conditioning_dimension(
  self : Self,
) -> Int {
  self.text_conditioning_dimension
}

///|
pub fn JointDiffusionDenoiserShape::timestep_frequency_dimension(
  self : Self,
) -> Int {
  self.timestep_frequency_dimension
}

///|
pub fn JointDiffusionDenoiserShape::timestep_embedding_hidden_dimension(
  self : Self,
) -> Int {
  self.timestep_embedding_hidden_dimension
}

///|
pub fn JointDiffusionDenoiserShape::final_adaln_conditioning_dimension(
  self : Self,
) -> Int {
  self.final_adaln_conditioning_dimension
}

///|
pub fn JointDiffusionDenoiserShape::final_norm_epsilon_numerator(
  self : Self,
) -> Int {
  self.final_norm_epsilon_numerator
}

///|
pub fn JointDiffusionDenoiserShape::final_norm_epsilon_denominator(
  self : Self,
) -> Int {
  self.final_norm_epsilon_denominator
}

///|
pub fn JointDiffusionDenoiserShape::final_norm_epsilon(self : Self) -> Double {
  self.final_norm_epsilon_numerator.to_double() /
  self.final_norm_epsilon_denominator.to_double()
}
