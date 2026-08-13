///|
/// Stable non-zero request seed for deterministic stochastic sampling.
///
/// The seed belongs to the transport-independent request contract so scheduler
/// and isolated workers can replay a token by `(seed, sample_index)` without
/// sharing mutable RNG state. It deliberately has no `Debug` implementation.
pub struct SamplingSeed {
  priv value : UInt64
} derive(Eq)

///|
pub fn SamplingSeed::new(
  value : UInt64,
) -> SamplingSeed raise InferenceContractError {
  if value == 0UL {
    raise invalid(SamplingSeed, OutOfRange, 0)
  }
  { value, }
}

///|
pub fn SamplingSeed::value(self : SamplingSeed) -> UInt64 {
  self.value
}
