///|
/// Pure numeric placement of the routed-output round, independent of device.
pub(all) enum MoeCombineRounding {
  SumThenBf16
  Bf16RoutedThenSum
} derive(Debug, Eq)

///|
pub extend MoeCombineRounding with Eq::{equal, not_equal}

///|
pub extend MoeCombineRounding with @debug.Debug::{to_repr}

///|
/// Two F32 inputs: globally reduced routed output and replicated shared
/// output. The latter is added once, never included in expert-rank reduction.
pub struct MoeCombinePrecision {
  priv maximum_rows : Int
  priv hidden : Int
  priv rounding : MoeCombineRounding
  priv input_bytes : Int64
  priv output_bytes : Int64
}

///|
pub fn MoeCombinePrecision::new(
  maximum_rows~ : Int,
  hidden~ : Int,
  rounding~ : MoeCombineRounding,
  memory_ceiling~ : Int64,
) -> Self raise PrecisionPlanError {
  guard maximum_rows > 0 && hidden > 0 else { raise IllegalPlacement }
  let cells = expert_checked_mul(maximum_rows.to_int64(), hidden.to_int64())
  let input_bytes = expert_checked_mul(cells, 4L)
  let output_bytes = expert_checked_mul(cells, 2L)
  let total = expert_checked_mul(cells, 10L)
  guard total <= memory_ceiling else { raise MemoryBudgetExceeded }
  { maximum_rows, hidden, rounding, input_bytes, output_bytes, }
}

///|
pub fn MoeCombinePrecision::geometry(self : Self) -> (Int, Int) {
  (self.maximum_rows, self.hidden)
}

///|
pub fn MoeCombinePrecision::rounding(self : Self) -> MoeCombineRounding {
  self.rounding
}

///|
pub fn MoeCombinePrecision::buffer_bytes(self : Self) -> (Int64, Int64) {
  (self.input_bytes, self.output_bytes)
}
