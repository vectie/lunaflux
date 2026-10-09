///|
/// Immutable text ingress/egress geometry. Selected output rows are distinct
/// from input rows: long prefill need not materialize a vocabulary per token.
/// Streams are replicated at ingress and averaged at egress. Mean and
/// normalized activations round to BF16 before BF16 affine multiplication.
pub struct TextIoPrecision {
  priv vocabulary : Int
  priv hidden : Int
  priv streams : Int
  priv rows : Int
  priv outputs : Int
  priv epsilon : Double
  priv residual_bytes : Int64
  priv normalized_bytes : Int64
  priv logits_bytes : Int64
  priv table_bytes : Int64
}

///|
pub fn TextIoPrecision::new(
  vocabulary_size~ : Int,
  hidden_width~ : Int,
  stream_count~ : Int,
  maximum_rows~ : Int,
  maximum_output_rows~ : Int,
  normalization_epsilon~ : Double,
  maximum_workspace_bytes~ : Int64,
  boundary? : TextBoundary = Complete,
) -> Self raise PrecisionPlanError {
  guard vocabulary_size > 0 &&
    hidden_width > 0 &&
    stream_count > 0 &&
    maximum_rows > 0 &&
    maximum_output_rows > 0 &&
    maximum_output_rows <= maximum_rows &&
    normalization_epsilon > 0.0 &&
    normalization_epsilon <= 1.0 &&
    !normalization_epsilon.is_nan() &&
    !normalization_epsilon.is_inf() else {
    raise IllegalPlacement
  }
  let residual_bytes = expert_checked_mul(
    expert_checked_mul(maximum_rows.to_int64(), hidden_width.to_int64()),
    expert_checked_mul(stream_count.to_int64(), 2L),
  )
  let normalized_bytes = expert_checked_mul(
    expert_checked_mul(maximum_output_rows.to_int64(), hidden_width.to_int64()),
    2L,
  )
  let logits_bytes = expert_checked_mul(
    expert_checked_mul(
      maximum_output_rows.to_int64(),
      vocabulary_size.to_int64(),
    ),
    4L,
  )
  let table_bytes = expert_checked_mul(
    expert_checked_mul(vocabulary_size.to_int64(), hidden_width.to_int64()),
    2L,
  )
  guard maximum_workspace_bytes >= 0L &&
    (
      !boundary.has_egress() ||
      (
        normalized_bytes <= maximum_workspace_bytes &&
        logits_bytes <= maximum_workspace_bytes - normalized_bytes
      )
    ) else {
    raise MemoryBudgetExceeded
  }
  {
    vocabulary: vocabulary_size,
    hidden: hidden_width,
    streams: stream_count,
    rows: maximum_rows,
    outputs: maximum_output_rows,
    epsilon: normalization_epsilon,
    residual_bytes,
    normalized_bytes,
    logits_bytes,
    table_bytes,
  }
}

///|
pub fn TextIoPrecision::geometry(self : Self) -> (Int, Int, Int, Int, Int) {
  (self.vocabulary, self.hidden, self.streams, self.rows, self.outputs)
}

///|
pub fn TextIoPrecision::normalization_epsilon(self : Self) -> Double {
  self.epsilon
}

///|
pub fn TextIoPrecision::residual_bytes(self : Self) -> Int64 {
  self.residual_bytes
}

///|
pub fn TextIoPrecision::normalized_bytes(self : Self) -> Int64 {
  self.normalized_bytes
}

///|
pub fn TextIoPrecision::logits_bytes(self : Self) -> Int64 {
  self.logits_bytes
}

///|
/// Each untied BF16 embedding/projection table has this size.
pub fn TextIoPrecision::table_bytes(self : Self) -> Int64 {
  self.table_bytes
}

///|
pub fn TextIoPrecision::workspace_bytes(self : Self) -> Int64 {
  self.normalized_bytes + self.logits_bytes
}
