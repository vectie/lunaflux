///|
/// A non-zero seed for the deterministic sampling stream.
///
/// Zero is rejected because it is the absorbing state of xorshift64. Seed
/// interpretation is part of the sampler's replay contract.
pub struct SamplingSeed {
  priv value : UInt64
} derive(Eq)

///|
pub fn SamplingSeed::new(value : UInt64) -> SamplingSeed raise SamplingError {
  if value == 0UL {
    raise InvalidSeed
  }
  { value, }
}

///|
pub fn SamplingSeed::value(self : SamplingSeed) -> UInt64 {
  self.value
}

///|
/// Mutable deterministic xorshift64 state owned by one sampling stream.
///
/// A successful stochastic selection advances the stream exactly once.
/// Validation failures do not advance it. Sharing one instance concurrently
/// is unsupported; schedulers should own one stream per request.
pub struct SamplingRng {
  priv mut state : UInt64
}

///|
pub fn SamplingRng::from_seed(seed : SamplingSeed) -> SamplingRng {
  { state: seed.value() }
}

///|
/// Returns the current replay checkpoint.
pub fn SamplingRng::state(self : SamplingRng) -> UInt64 {
  self.state
}

///|
/// Replaces the stream state with a validated non-zero checkpoint.
pub fn SamplingRng::reseed(self : SamplingRng, seed : SamplingSeed) -> Unit {
  self.state = seed.value()
}

///|
fn SamplingRng::next_unit(self : SamplingRng) -> Double {
  let mut value = self.state
  value = value ^ (value << 13)
  value = value ^ (value >> 7)
  value = value ^ (value << 17)
  self.state = value
  // The upper 53 bits are exactly representable as a Double. Multiplication
  // by 2^-53 produces a value in [0, 1) without rounding up to one.
  (value >> 11).to_double() * 1.1102230246251565e-16
}
