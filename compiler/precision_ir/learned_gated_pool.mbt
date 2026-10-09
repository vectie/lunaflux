///|
/// Per-feature learned gated pooling, with optional preceding-window overlap.
/// BF16 input and parameters project in F32; pooling publishes BF16 before
/// learned F32 RMS normalization. Rotation/cache quantization are later passes.
pub struct LearnedGatedPoolPrecision {
  priv input_width : Int
  priv feature_width : Int
  priv ratio : Int
  priv overlap : Bool
  priv rows : Int
  priv positions : Int
  priv epsilon : Double
  priv states : ReadOnlyArray[Int64]
  priv frames : ReadOnlyArray[Int64]
  priv parameters : ReadOnlyArray[Int64]
}

///|
pub fn LearnedGatedPoolPrecision::new(
  input_width~ : Int,
  feature_width~ : Int,
  ratio~ : Int,
  preceding_window_overlap~ : Bool,
  maximum_rows~ : Int,
  maximum_positions~ : Int,
  norm_epsilon~ : Double,
  maximum_state_bytes~ : Int64,
  maximum_workspace_bytes~ : Int64,
) -> Self raise PrecisionPlanError {
  guard input_width > 0 &&
    input_width <= 65536 &&
    feature_width > 0 &&
    feature_width <= 2048 &&
    ratio > 0 &&
    ratio <= 256 &&
    maximum_rows > 0 &&
    maximum_rows <= 65536 &&
    maximum_positions >= maximum_rows &&
    maximum_positions <= 1048576 &&
    norm_epsilon > 0.0 &&
    !norm_epsilon.is_nan() &&
    !norm_epsilon.is_inf() else {
    raise IllegalPlacement
  }
  let coefficient = if preceding_window_overlap { 2L } else { 1L }
  let channels = coefficient * feature_width.to_int64()
  let state = coefficient * ratio.to_int64() * channels * 4L
  let compressed_rows = (maximum_rows + ratio - 1) / ratio
  let states = [state, state, 8L]
  let frames = [
    maximum_rows.to_int64() * channels * 4L,
    maximum_rows.to_int64() * channels * 4L,
    compressed_rows.to_int64() * feature_width.to_int64() * 2L,
    compressed_rows.to_int64() * feature_width.to_int64() * 2L,
    compressed_rows.to_int64() * 4L,
    20L,
    24L,
  ]
  guard states.fold(init=0L, (total, bytes) => total + bytes) <=
    maximum_state_bytes &&
    frames.fold(init=0L, (total, bytes) => total + bytes) <=
    maximum_workspace_bytes else {
    raise MemoryBudgetExceeded
  }
  {
    input_width,
    feature_width,
    ratio,
    overlap: preceding_window_overlap,
    rows: maximum_rows,
    positions: maximum_positions,
    epsilon: norm_epsilon,
    states: ReadOnlyArray::from_array(states),
    frames: ReadOnlyArray::from_array(frames),
    parameters: ReadOnlyArray::from_array([
      channels * input_width.to_int64() * 2L,
      channels * input_width.to_int64() * 2L,
      ratio.to_int64() * channels * 4L,
      feature_width.to_int64() * 2L,
    ]),
  }
}

///|
pub fn LearnedGatedPoolPrecision::geometry(
  self : Self,
) -> (Int, Int, Int, Bool, Int, Int) {
  (
    self.input_width,
    self.feature_width,
    self.ratio,
    self.overlap,
    self.rows,
    self.positions,
  )
}

///|
pub fn LearnedGatedPoolPrecision::norm_epsilon(self : Self) -> Double {
  self.epsilon
}

///|
pub fn LearnedGatedPoolPrecision::compressed_rows(self : Self) -> Int {
  (self.rows + self.ratio - 1) / self.ratio
}

///|
pub fn LearnedGatedPoolPrecision::state_bytes(
  self : Self,
) -> ReadOnlyArray[Int64] {
  self.states
}

///|
pub fn LearnedGatedPoolPrecision::frame_bytes(
  self : Self,
) -> ReadOnlyArray[Int64] {
  self.frames
}

///|
pub fn LearnedGatedPoolPrecision::parameter_bytes(
  self : Self,
) -> ReadOnlyArray[Int64] {
  self.parameters
}

///|
pub fn LearnedGatedPoolPrecision::effect(_self : Self) -> NumericEffect {
  CacheCommit
}
