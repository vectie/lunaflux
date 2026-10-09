///|
/// Request-owned immutable-position row storage, with exact 16-bit cell copies.
/// `position_stride` maps an absolute source position to a compact row ordinal.
/// A zero logical capacity has one physical sentinel row, never a readable row.
/// CUDA submission geometry and model/compression names do not belong here.
pub struct RetainedRowsPrecision {
  priv width : Int
  priv stride : Int
  priv rows : Int
  priv capacity : Int
  priv states : ReadOnlyArray[Int64]
}

///|
pub fn RetainedRowsPrecision::new(
  word_width~ : Int,
  position_stride~ : Int,
  maximum_input_rows~ : Int,
  maximum_stored_rows~ : Int,
  maximum_state_bytes~ : Int64,
  maximum_workspace_bytes~ : Int64,
) -> Self raise PrecisionPlanError {
  guard word_width > 0 &&
    word_width <= 65536 &&
    position_stride > 0 &&
    position_stride <= 1048576 &&
    maximum_input_rows > 0 &&
    maximum_input_rows <= 65536 &&
    maximum_stored_rows >= 0 &&
    maximum_stored_rows <= 1048576 / position_stride else {
    raise IllegalPlacement
  }
  // Raw words; [length,error]; conventional five-word count port. The length
  // is published only after all copies, and every failure exposes zero rows.
  let states = [
    maximum_stored_rows.max(1).to_int64() * word_width.to_int64() * 2L,
    8L,
    20L,
  ]
  guard states.fold(init=0L, (total, bytes) => total + bytes) <=
    maximum_state_bytes &&
    maximum_workspace_bytes >= 12L else {
    raise MemoryBudgetExceeded
  }
  {
    width: word_width,
    stride: position_stride,
    rows: maximum_input_rows,
    capacity: maximum_stored_rows,
    states: ReadOnlyArray::from_array(states),
  }
}

///|
pub fn RetainedRowsPrecision::geometry(self : Self) -> (Int, Int, Int, Int) {
  (self.width, self.stride, self.rows, self.capacity)
}

///|
pub fn RetainedRowsPrecision::state_bytes(self : Self) -> ReadOnlyArray[Int64] {
  self.states
}

///|
pub fn RetainedRowsPrecision::workspace_bytes(_self : Self) -> Int64 {
  12L
}

///|
pub fn RetainedRowsPrecision::effect(_self : Self) -> NumericEffect {
  CacheCommit
}
