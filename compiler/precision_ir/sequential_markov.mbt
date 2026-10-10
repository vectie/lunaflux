///|
/// Immutable dependent categorical chain. Row i reads token i, gathers its
/// BF16 embedding, adds an F32 projected bias to row i's F32 logits, then
/// samples token i+1. Confidence uses the same previous-token embedding and
/// pre-normalization BF16 hidden row, with an F32 dot and no sigmoid.
/// Temperature zero is stable argmax; positive temperature is categorical
/// Gumbel-max. RNG seed and sequence are explicit caller-owned device ports.
pub struct SequentialMarkovPrecision {
  priv vocabulary : Int
  priv hidden : Int
  priv rank : Int
  priv rows : Int
  priv temperature : Double
  priv embedding_bytes : Int64
  priv confidence_weight_bytes : Int64
  priv workspace_bytes : Int64
}

///|
pub fn SequentialMarkovPrecision::new(
  vocabulary_size~ : Int,
  hidden_width~ : Int,
  embedding_width~ : Int,
  maximum_rows~ : Int,
  temperature~ : Double,
  maximum_workspace_bytes~ : Int64,
) -> Self raise PrecisionPlanError {
  guard vocabulary_size > 0 &&
    hidden_width > 0 &&
    embedding_width > 0 &&
    maximum_rows > 0 &&
    maximum_rows <= 1048576 &&
    temperature >= 0.0 &&
    !temperature.is_nan() &&
    !temperature.is_inf() else {
    raise IllegalPlacement
  }
  let embedding_bytes = expert_checked_mul(
    expert_checked_mul(vocabulary_size.to_int64(), embedding_width.to_int64()),
    2L,
  )
  let confidence_weight_bytes = expert_checked_mul(
    hidden_width.to_int64() + embedding_width.to_int64(),
    2L,
  )
  let embedding_frame_bytes = expert_checked_mul(
    expert_checked_mul(maximum_rows.to_int64(), embedding_width.to_int64()),
    2L,
  )
  let workspace_bytes = expert_checked_mul(maximum_rows.to_int64(), 4L) +
    embedding_frame_bytes
  guard workspace_bytes <= maximum_workspace_bytes else {
    raise MemoryBudgetExceeded
  }
  {
    vocabulary: vocabulary_size,
    hidden: hidden_width,
    rank: embedding_width,
    rows: maximum_rows,
    temperature,
    embedding_bytes,
    confidence_weight_bytes,
    workspace_bytes,
  }
}

///|
pub fn SequentialMarkovPrecision::geometry(self : Self) -> (Int, Int, Int, Int) {
  (self.vocabulary, self.hidden, self.rank, self.rows)
}

///|
pub fn SequentialMarkovPrecision::temperature(self : Self) -> Double {
  self.temperature
}

///|
/// Gather table and F32-evaluated projection are both stored as BF16.
pub fn SequentialMarkovPrecision::weight_bytes(
  self : Self,
) -> (Int64, Int64, Int64) {
  (self.embedding_bytes, self.embedding_bytes, self.confidence_weight_bytes)
}

///|
pub fn SequentialMarkovPrecision::workspace_bytes(self : Self) -> Int64 {
  self.workspace_bytes
}

///|
pub fn SequentialMarkovPrecision::embedding_frame_bytes(self : Self) -> Int64 {
  self.rows.to_int64() * self.rank.to_int64() * 2L
}

///|
pub fn SequentialMarkovPrecision::hidden_bytes(self : Self) -> Int64 {
  self.rows.to_int64() * self.hidden.to_int64() * 2L
}

///|
pub fn SequentialMarkovPrecision::logits_bytes(
  self : Self,
) -> Int64 raise PrecisionPlanError {
  expert_checked_mul(
    expert_checked_mul(self.rows.to_int64(), self.vocabulary.to_int64()),
    4L,
  )
}
