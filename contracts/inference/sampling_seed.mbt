///|
/// Stable non-zero request seed for deterministic stochastic sampling.
///
/// The seed belongs to the transport-independent request contract so scheduler
/// and isolated workers can replay a token by `(seed, sample_index)` without
/// sharing mutable RNG state. It deliberately has no `Debug` implementation.
#valtype
pub struct SamplingSeed {
  priv value : UInt64
} derive(Eq)

///|
pub fn SamplingSeed::new(
  value : UInt64,
) -> SamplingSeed raise InferenceContractError {
  let sampling_seed : SamplingSeed = { value, }
  sampling_seed.validate()
  sampling_seed
}

///|
pub fn SamplingSeed::value(self : SamplingSeed) -> UInt64 {
  self.value
}

///|
fn SamplingSeed::validate(
  self : SamplingSeed,
) -> Unit raise InferenceContractError {
  if self.value == 0UL {
    raise invalid(SamplingSeed, OutOfRange, 0)
  }
}
