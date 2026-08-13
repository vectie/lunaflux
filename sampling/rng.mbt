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
pub fn SamplingRng::from_seed(seed : @inference.SamplingSeed) -> SamplingRng {
  { state: seed.value() }
}

///|
/// Returns the current replay checkpoint.
pub fn SamplingRng::state(self : SamplingRng) -> UInt64 {
  self.state
}

///|
/// Replaces the stream state with a validated non-zero checkpoint.
pub fn SamplingRng::reseed(
  self : SamplingRng,
  seed : @inference.SamplingSeed,
) -> Unit {
  self.state = seed.value()
}

///|
fn counter_state(seed : @inference.SamplingSeed, sample_index : Int) -> UInt64 {
  let mut value = seed.value() +
    (sample_index + 1).to_uint64() * 0x9e3779b97f4a7c15UL
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9UL
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebUL
  value = value ^ (value >> 31)
  if value == 0UL {
    0x9e3779b97f4a7c15UL
  } else {
    value
  }
}

///|
fn counter_unit(seed : @inference.SamplingSeed, sample_index : Int) -> Double {
  let value = counter_state(seed, sample_index)
  (value >> 11).to_double() * 1.1102230246251565e-16
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
