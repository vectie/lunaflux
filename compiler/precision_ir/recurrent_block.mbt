///|
/// Complete bias-free recurrent attention branch. Projections have ordered
/// F32 sums and explicit BF16 stage rounds; decay and cache remain F32.
/// The plan describes arithmetic and storage, never a device schedule.
pub struct RecurrentBlockPrecision {
  priv delta : RecurrentDeltaPrecision
  priv convolution : CausalConvolutionPrecision
  priv hidden : Int
  priv rank : Int
  priv epsilon : Double
  priv frame_bytes : Int64
}

///|
pub fn RecurrentBlockPrecision::new(
  delta : RecurrentDeltaPrecision,
  convolution : CausalConvolutionPrecision,
  hidden_width~ : Int,
  control_rank~ : Int,
  output_normalization_epsilon~ : Double,
) -> Self raise PrecisionPlanError {
  let (heads, keys, values) = delta.geometry()
  let (channels, _) = convolution.geometry()
  guard hidden_width > 0 &&
    control_rank > 0 &&
    keys == values &&
    heads.to_int64() * keys.to_int64() == channels.to_int64() &&
    delta.capacity() == convolution.capacity() &&
    !output_normalization_epsilon.is_nan() &&
    !output_normalization_epsilon.is_inf() &&
    output_normalization_epsilon > 0.0 &&
    output_normalization_epsilon <= 1.0 else {
    raise IllegalPlacement
  }
  let (rows, _, _) = delta.capacity()
  ignore(
    expert_checked_mul(
      expert_checked_mul(rows.to_int64(), hidden_width.to_int64()),
      2L,
    ),
  )
  // Q/K/V, raw forget, F32 decay (two BF16-sized frames), output gate,
  // recurrent output, normalized output, two low-rank frames and two beta frames.
  let width = 9L * channels.to_int64() +
    2L * control_rank.to_int64() +
    2L * heads.to_int64()
  let frame_bytes = expert_checked_mul(
    expert_checked_mul(rows.to_int64(), width),
    2L,
  )
  ignore(
    expert_checked_mul(
      expert_checked_mul(channels.to_int64(), hidden_width.to_int64()),
      2L,
    ),
  )
  ignore(
    expert_checked_mul(
      expert_checked_mul(control_rank.to_int64(), hidden_width.to_int64()),
      2L,
    ),
  )
  ignore(
    expert_checked_mul(
      expert_checked_mul(channels.to_int64(), control_rank.to_int64()),
      2L,
    ),
  )
  {
    delta,
    convolution,
    hidden: hidden_width,
    rank: control_rank,
    epsilon: output_normalization_epsilon,
    frame_bytes,
  }
}

///|
pub fn RecurrentBlockPrecision::delta(self : Self) -> RecurrentDeltaPrecision {
  self.delta
}

///|
pub fn RecurrentBlockPrecision::convolution(
  self : Self,
) -> CausalConvolutionPrecision {
  self.convolution
}

///|
pub fn RecurrentBlockPrecision::projection_geometry(self : Self) -> (Int, Int) {
  (self.hidden, self.rank)
}

///|
pub fn RecurrentBlockPrecision::normalization_epsilon(self : Self) -> Double {
  self.epsilon
}

///|
pub fn RecurrentBlockPrecision::frame_bytes(self : Self) -> Int64 {
  self.frame_bytes
}
