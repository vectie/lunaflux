///|
pub(all) enum ConvolutionProjectionBoundary {
  Float32BeforeSilu
  Bf16BeforeSilu
}

///|
/// Request-owned depthwise causal convolution. BF16 input, weights and history;
/// ordered F32 accumulation, explicit pre-SiLU boundary, BF16 output. History stores
/// raw inputs, never activated outputs, oldest lag first.
pub struct CausalConvolutionPrecision {
  priv channels : Int32
  priv kernel : Int32
  priv rows : Int32
  priv sequences : Int32
  priv slots : Int32
  priv state_bytes : Int64
  priv frame_bytes : Int64
  priv projection_boundary : ConvolutionProjectionBoundary
}

///|
pub fn CausalConvolutionPrecision::new(
  channel_count~ : Int32,
  kernel_size~ : Int32,
  maximum_rows~ : Int32,
  maximum_sequences~ : Int32,
  cache_slots~ : Int32,
  maximum_state_bytes~ : Int64,
  projection_boundary~ : ConvolutionProjectionBoundary = Float32BeforeSilu,
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
    projection_boundary,
  }
}

///|
pub fn CausalConvolutionPrecision::projection_boundary(
  self : Self,
) -> ConvolutionProjectionBoundary {
  self.projection_boundary
}

///|
pub fn CausalConvolutionPrecision::geometry(self : Self) -> (Int32, Int32) {
  (self.channels, self.kernel)
}

///|
pub fn CausalConvolutionPrecision::capacity(
  self : Self,
) -> (Int32, Int32, Int32) {
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
