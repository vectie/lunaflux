///|
/// Request-owned depthwise causal convolution. BF16 input, weights and history;
/// ordered F32 accumulation, SiLU, one BF16 output rounding. History stores
/// raw inputs, never activated outputs, oldest lag first.
pub struct CausalConvolutionPrecision {
  priv channels : Int
  priv kernel : Int
  priv rows : Int
  priv sequences : Int
  priv slots : Int
  priv state_bytes : Int64
  priv frame_bytes : Int64
}

///|
pub fn CausalConvolutionPrecision::new(
  channel_count~ : Int,
  kernel_size~ : Int,
  maximum_rows~ : Int,
  maximum_sequences~ : Int,
  cache_slots~ : Int,
  maximum_state_bytes~ : Int64,
) -> Self raise PrecisionPlanError {
  guard channel_count > 0 &&
    kernel_size > 1 &&
    maximum_rows > 0 &&
    maximum_sequences > 0 &&
    maximum_sequences <= maximum_rows &&
    cache_slots >= maximum_sequences else {
    raise IllegalPlacement
  }
  let state_bytes = expert_checked_mul(
    expert_checked_mul(
      expert_checked_mul(cache_slots.to_int64(), channel_count.to_int64()),
      (kernel_size - 1).to_int64(),
    ),
    2L,
  )
  guard state_bytes <= maximum_state_bytes else { raise MemoryBudgetExceeded }
  let frame_bytes = expert_checked_mul(
    expert_checked_mul(maximum_rows.to_int64(), channel_count.to_int64()),
    2L,
  )
  ignore(
    expert_checked_mul(
      expert_checked_mul(channel_count.to_int64(), kernel_size.to_int64()),
      2L,
    ),
  )
  {
    channels: channel_count,
    kernel: kernel_size,
    rows: maximum_rows,
    sequences: maximum_sequences,
    slots: cache_slots,
    state_bytes,
    frame_bytes,
  }
}

///|
pub fn CausalConvolutionPrecision::geometry(self : Self) -> (Int, Int) {
  (self.channels, self.kernel)
}

///|
pub fn CausalConvolutionPrecision::capacity(self : Self) -> (Int, Int, Int) {
  (self.rows, self.sequences, self.slots)
}

///|
pub fn CausalConvolutionPrecision::state_bytes(self : Self) -> Int64 {
  self.state_bytes
}

///|
pub fn CausalConvolutionPrecision::frame_bytes(self : Self) -> Int64 {
  self.frame_bytes
}

///|
pub fn CausalConvolutionPrecision::effect(self : Self) -> NumericEffect {
  ignore(self)
  CacheCommit
}
