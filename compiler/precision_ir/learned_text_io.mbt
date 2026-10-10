///|
/// Learned residual-stream collapse followed by the ordinary text egress.
/// Control projection accumulates flattened BF16 streams in F32, then scales
/// the dot by inverse RMS. Final RMSNorm rounds only after weight multiplication;
/// the vocabulary projection publishes F32 logits without an extra BF16 cast.
/// sigmoid controls weight each stream and collapse rounds once to BF16.
/// No model names or device schedule enter this immutable numerical plan.
pub struct LearnedTextIoPrecision {
  priv text : TextIoPrecision
  priv streams : Int
  priv control_epsilon : Double
  priv control_normalization_epsilon : Double
  priv input_bytes : Int64
  priv collapse_bytes : Int64
  priv function_bytes : Int64
  priv base_bytes : Int64
  priv workspace_bytes : Int64
}

///|
pub fn LearnedTextIoPrecision::new(
  vocabulary_size~ : Int,
  hidden_width~ : Int,
  stream_count~ : Int,
  maximum_rows~ : Int,
  maximum_output_rows~ : Int,
  normalization_epsilon~ : Double,
  control_epsilon~ : Double,
  control_normalization_epsilon~ : Double,
  maximum_workspace_bytes~ : Int64,
) -> Self raise PrecisionPlanError {
  guard stream_count > 1 &&
    stream_count <= 16 &&
    hidden_width > 0 &&
    hidden_width <= 65536 &&
    maximum_rows > 0 &&
    maximum_rows <= 1048576 &&
    control_epsilon > 0.0 &&
    control_epsilon <= 1.0 &&
    !control_epsilon.is_nan() &&
    !control_epsilon.is_inf() &&
    control_normalization_epsilon > 0.0 &&
    control_normalization_epsilon <= 1.0 &&
    !control_normalization_epsilon.is_nan() &&
    !control_normalization_epsilon.is_inf() else {
    raise IllegalPlacement
  }
  let text = TextIoPrecision::new(
    vocabulary_size~,
    hidden_width~,
    stream_count=1,
    maximum_rows~,
    maximum_output_rows~,
    normalization_epsilon~,
    maximum_workspace_bytes~,
    boundary=Egress,
    normalization_law=F32NormalizeThenWeight,
    logits_law=F32Accumulate,
  )
  let collapse_bytes = text.residual_bytes()
  let input_bytes = expert_checked_mul(collapse_bytes, stream_count.to_int64())
  let function_bytes = expert_checked_mul(
    expert_checked_mul(stream_count.to_int64(), stream_count.to_int64()),
    expert_checked_mul(hidden_width.to_int64(), 4L),
  )
  let base_bytes = stream_count.to_int64() * 4L
  guard collapse_bytes <= maximum_workspace_bytes - text.workspace_bytes() else {
    raise MemoryBudgetExceeded
  }
  {
    text,
    streams: stream_count,
    control_epsilon,
    control_normalization_epsilon,
    input_bytes,
    collapse_bytes,
    function_bytes,
    base_bytes,
    workspace_bytes: collapse_bytes + text.workspace_bytes(),
  }
}

///|
pub fn LearnedTextIoPrecision::text(self : Self) -> TextIoPrecision {
  self.text
}

///|
pub fn LearnedTextIoPrecision::stream_count(self : Self) -> Int {
  self.streams
}

///|
pub fn LearnedTextIoPrecision::control_epsilon(self : Self) -> Double {
  self.control_epsilon
}

///|
pub fn LearnedTextIoPrecision::control_normalization_epsilon(
  self : Self,
) -> Double {
  self.control_normalization_epsilon
}

///|
pub fn LearnedTextIoPrecision::input_bytes(self : Self) -> Int64 {
  self.input_bytes
}

///|
pub fn LearnedTextIoPrecision::collapse_bytes(self : Self) -> Int64 {
  self.collapse_bytes
}

///|
/// Function matrix, base vector and scalar, all F32 and in this order.
pub fn LearnedTextIoPrecision::control_weight_bytes(
  self : Self,
) -> (Int64, Int64, Int64) {
  (self.function_bytes, self.base_bytes, 4L)
}

///|
pub fn LearnedTextIoPrecision::workspace_bytes(self : Self) -> Int64 {
  self.workspace_bytes
}
