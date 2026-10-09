///|
/// Pure composition of the sparse branch and its residual-stream envelope.
/// Expanded attention output is an explicit intermediate; output projection
/// geometry is independent of the sparse index and mHC ownership.
pub struct IndexedSparseDecoderPrecision {
  priv projection : IndexedSparseProjectionPrecision
  priv history : IndexedSparseHistoryPrecision
  priv envelope : HyperConnectionPrecision
  priv attention_output_bytes : Int64
  priv output_weight_bytes : Int64
  priv workspace : Int64
}

///|
pub fn IndexedSparseDecoderPrecision::new(
  projection : IndexedSparseProjectionPrecision,
  envelope : HyperConnectionPrecision,
  maximum_state_bytes~ : Int64,
  maximum_workspace_bytes~ : Int64,
) -> Self raise PrecisionPlanError {
  let attention = projection.attention()
  let (rows, _) = attention.capacity()
  let (hidden, _, _) = projection.geometry()
  let (_, envelope_hidden, envelope_rows) = envelope.geometry()
  guard hidden == envelope_hidden && rows == envelope_rows else {
    raise IllegalPlacement
  }
  let (heads, _, _, values) = attention.attention_geometry()
  let width = expert_checked_mul(heads.to_int64(), values.to_int64())
  let output = expert_checked_mul(
    expert_checked_mul(rows.to_int64(), width),
    2L,
  )
  guard output <= 4294967294L else { raise ArithmeticOverflow }
  let weight = expert_checked_mul(
    expert_checked_mul(hidden.to_int64(), width),
    2L,
  )
  let workspace = for
    size in [
      projection.workspace_bytes(),
      attention.workspace_bytes(),
      envelope.workspace_bytes(),
      output,
    ]
    total = 0L {
    guard size <= 9223372036854775807L - total else { raise ArithmeticOverflow }
    continue total + size
  } nobreak {
    total
  }
  guard workspace <= maximum_workspace_bytes else { raise MemoryBudgetExceeded }
  {
    projection,
    history: IndexedSparseHistoryPrecision::new(attention, maximum_state_bytes~),
    envelope,
    attention_output_bytes: output,
    output_weight_bytes: weight,
    workspace,
  }
}

///|
pub fn IndexedSparseDecoderPrecision::projection(
  self : Self,
) -> IndexedSparseProjectionPrecision {
  self.projection
}

///|
pub fn IndexedSparseDecoderPrecision::history(
  self : Self,
) -> IndexedSparseHistoryPrecision {
  self.history
}

///|
pub fn IndexedSparseDecoderPrecision::envelope(
  self : Self,
) -> HyperConnectionPrecision {
  self.envelope
}

///|
pub fn IndexedSparseDecoderPrecision::attention_output_bytes(
  self : Self,
) -> Int64 {
  self.attention_output_bytes
}

///|
pub fn IndexedSparseDecoderPrecision::output_weight_bytes(self : Self) -> Int64 {
  self.output_weight_bytes
}

///|
pub fn IndexedSparseDecoderPrecision::workspace_bytes(self : Self) -> Int64 {
  self.workspace
}
