///|
/// Fail-closed errors from the exact Flash-0731 correctness fixture.
pub(all) suberror DeepSeekV4ReferenceFixtureError {
  UnsupportedFixtureProfile
  InvalidFixturePlan
  ReferenceFailure(@advanced_decoder_reference.AdvancedDecoderReferenceError)
} derive(Debug, Eq)

///|
/// Immutable observations from one bounded DeepSeek-V4 reference fixture.
/// The values are deliberately tiny; they prove plan interpretation rather
/// than checkpoint-scale numerical execution.
pub struct DeepSeekV4ReferenceFixtureResult {
  priv plan_digest : @spec.PlanDigest
  priv top_k_routing : @advanced_decoder_reference.ExpertRouting
  priv hash_routing : @advanced_decoder_reference.ExpertRouting
  priv expert_output : ReadOnlyArray[Float]
  priv learned_indices : @advanced_decoder_reference.ReferenceTopK
  priv compressed_indices : ReadOnlyArray[Int]
  priv mixed_streams : ReadOnlyArray[Float]
  priv auxiliary_stages : ReadOnlyArray[
    @advanced_decoder_reference.AuxiliaryReferenceStage,
  ]
} derive(Debug, Eq)

///|
pub fn DeepSeekV4ReferenceFixtureResult::plan_digest(
  self : Self,
) -> @spec.PlanDigest {
  self.plan_digest
}

///|
pub fn DeepSeekV4ReferenceFixtureResult::top_k_routing(
  self : Self,
) -> @advanced_decoder_reference.ExpertRouting {
  self.top_k_routing
}

///|
pub fn DeepSeekV4ReferenceFixtureResult::hash_routing(
  self : Self,
) -> @advanced_decoder_reference.ExpertRouting {
  self.hash_routing
}

///|
pub fn DeepSeekV4ReferenceFixtureResult::expert_output(
  self : Self,
) -> ReadOnlyArray[Float] {
  self.expert_output
}

///|
pub fn DeepSeekV4ReferenceFixtureResult::learned_indices(
  self : Self,
) -> @advanced_decoder_reference.ReferenceTopK {
  self.learned_indices
}

///|
pub fn DeepSeekV4ReferenceFixtureResult::compressed_indices(
  self : Self,
) -> ReadOnlyArray[Int] {
  self.compressed_indices
}

///|
pub fn DeepSeekV4ReferenceFixtureResult::mixed_streams(
  self : Self,
) -> ReadOnlyArray[Float] {
  self.mixed_streams
}

///|
pub fn DeepSeekV4ReferenceFixtureResult::auxiliary_stages(
  self : Self,
) -> ReadOnlyArray[@advanced_decoder_reference.AuxiliaryReferenceStage] {
  self.auxiliary_stages
}
