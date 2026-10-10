///|
/// Physical arithmetic contracts for residual stream combination. These are
/// numerical alternatives, not model names or device scheduling choices.
/// RoundedBf16ControlsAndProducts consumes combination[source,destination]
/// (the residual fold is transposed); OrderedF32SingleBf16Round consumes
/// combination[destination,source]. They are distinct numerical contracts,
/// not interchangeable performance schedules.
/// TransposedF32ResidualThenBranch consumes combination[source,destination],
/// accumulates the residual independently in F32, then adds the F32 branch
/// product and rounds once to BF16.
pub(all) enum HyperConnectionMixLaw {
  RoundedBf16ControlsAndProducts
  OrderedF32SingleBf16Round
  TransposedF32ResidualThenBranch
}

///|
/// Storage and rounding order for learned control projection. These are
/// numerical contracts, not model-family or hardware alternatives.
pub(all) enum HyperConnectionControlLaw {
  NormalizedBf16Projection
  F32ProjectionThenNormalize
}

///|
/// Immutable mHC branch envelope: controls, positive Sinkhorn, collapsed
/// branch input, weighted RMSNorm, and residual-stream publication.
pub struct HyperConnectionPrecision {
  priv streams : Int
  priv hidden : Int
  priv rows : Int
  priv iterations : Int
  priv control_epsilon : Double
  priv normalization_epsilon : Double
  priv law : HyperConnectionMixLaw
  priv control_law : HyperConnectionControlLaw
  priv function_bytes : Int64
  priv frames : ReadOnlyArray[Int64]
  priv bytes : Int64
}

///|
pub fn HyperConnectionPrecision::new(
  stream_count~ : Int,
  hidden_width~ : Int,
  maximum_rows~ : Int,
  sinkhorn_iterations~ : Int,
  control_epsilon~ : Double,
  normalization_epsilon~ : Double,
  mix_law~ : HyperConnectionMixLaw,
  control_law? : HyperConnectionControlLaw = NormalizedBf16Projection,
) -> Self raise PrecisionPlanError {
  guard stream_count > 1 &&
    hidden_width > 0 &&
    maximum_rows > 0 &&
    sinkhorn_iterations > 0 &&
    !control_epsilon.is_nan() &&
    !control_epsilon.is_inf() &&
    control_epsilon > 0.0 &&
    control_epsilon <= 1.0 &&
    !normalization_epsilon.is_nan() &&
    !normalization_epsilon.is_inf() &&
    normalization_epsilon > 0.0 &&
    normalization_epsilon <= 1.0 else {
    raise IllegalPlacement
  }
  let rows = maximum_rows.to_int64()
  let streams = stream_count.to_int64()
  let hidden = hidden_width.to_int64()
  let vector = expert_checked_mul(expert_checked_mul(rows, streams), 4L)
  let matrix = expert_checked_mul(vector, streams)
  let branch = expert_checked_mul(expert_checked_mul(rows, hidden), 2L)
  ignore(expert_checked_mul(branch, streams))
  // Prove control-projection weight addressing, not just frame allocation.
  let function_bytes = expert_checked_mul(
    expert_checked_mul(expert_checked_mul(streams + 2L, streams), streams),
    expert_checked_mul(
      hidden,
      match control_law {
        NormalizedBf16Projection => 2L
        F32ProjectionThenNormalize => 4L
      },
    ),
  )
  let frames = [vector, matrix, matrix, branch, branch, branch]
  let bytes = for size in frames; total = 0L {
    guard size <= 9223372036854775807L - total else {
      raise MemoryBudgetExceeded
    }
    continue total + size
  } nobreak {
    total
  }
  {
    streams: stream_count,
    hidden: hidden_width,
    rows: maximum_rows,
    iterations: sinkhorn_iterations,
    control_epsilon,
    normalization_epsilon,
    law: mix_law,
    control_law,
    function_bytes,
    frames: ReadOnlyArray::from_array(frames),
    bytes,
  }
}

///|
pub fn HyperConnectionPrecision::geometry(self : Self) -> (Int, Int, Int) {
  (self.streams, self.hidden, self.rows)
}

///|
pub fn HyperConnectionPrecision::sinkhorn_iterations(self : Self) -> Int {
  self.iterations
}

///|
pub fn HyperConnectionPrecision::control_epsilon(self : Self) -> Double {
  self.control_epsilon
}

///|
pub fn HyperConnectionPrecision::normalization_epsilon(self : Self) -> Double {
  self.normalization_epsilon
}

///|
pub fn HyperConnectionPrecision::mix_law(self : Self) -> HyperConnectionMixLaw {
  self.law
}

///|
pub fn HyperConnectionPrecision::control_law(
  self : Self,
) -> HyperConnectionControlLaw {
  self.control_law
}

///|
pub fn HyperConnectionPrecision::function_weight_bytes(self : Self) -> Int64 {
  self.function_bytes
}

///|
/// post, initial matrix, normalized matrix, collapse, normalized input, branch.
pub fn HyperConnectionPrecision::frame_bytes(
  self : Self,
) -> ReadOnlyArray[Int64] {
  self.frames
}

///|
pub fn HyperConnectionPrecision::workspace_bytes(self : Self) -> Int64 {
  self.bytes
}
