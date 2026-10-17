///|
/// Two byte-preserving row views share storage across disjoint consumer epochs.
/// Routing/publication keep original job order; only operand storage is padded.
pub struct PackedExpertRows {
  priv rows : Int
  priv base : Int64
  priv payloads : (Int, Int)
  priv scales : (Int, Int)
  priv bytes : Int64
}

///|
pub fn PackedExpertRows::new(
  grouping : ExpertGroupingPlan,
  after_bytes~ : Int64,
  payload_bytes~ : (Int, Int),
  scale_bytes~ : (Int, Int),
) -> Self raise @precision.PrecisionPlanError {
  guard after_bytes >= 0L &&
    payload_bytes.0 > 0 &&
    payload_bytes.1 > 0 &&
    scale_bytes.0 > 0 &&
    scale_bytes.1 > 0 else {
    raise @precision.PrecisionPlanError::MemoryBudgetExceeded
  }
  let rows = grouping.maximum_groups() * grouping.rows_per_group()
  let first = rows.to_int64() *
    (payload_bytes.0.to_int64() + scale_bytes.0.to_int64())
  let second = rows.to_int64() *
    (payload_bytes.1.to_int64() + scale_bytes.1.to_int64())
  let bytes = (Int64::max(first, second) + 15L) / 16L * 16L
  guard after_bytes <= 9223372036854775792L - bytes else {
    raise @precision.PrecisionPlanError::MemoryBudgetExceeded
  }
  {
    rows,
    base: (after_bytes + 15L) / 16L * 16L,
    payloads: payload_bytes,
    scales: scale_bytes,
    bytes,
  }
}

///|
pub fn PackedExpertRows::base(self : Self) -> Int64 {
  self.base
}

///|
pub fn PackedExpertRows::total_bytes(self : Self) -> Int64 {
  self.base + self.bytes
}

///|
pub fn PackedExpertRows::plane(
  self : Self,
  product : Bool,
) -> (Int, Int, Int64) {
  let payload = if product { self.payloads.1 } else { self.payloads.0 }
  let scale = if product { self.scales.1 } else { self.scales.0 }
  (payload, scale, self.rows.to_int64() * payload.to_int64())
}

///|
pub fn PackedExpertRows::row(self : Self, group : Int, within : Int) -> Int? {
  guard group >= 0 && group < self.rows / 16 && within >= 0 && within < 16 else {
    return None
  }
  Some(group * 16 + within)
}
