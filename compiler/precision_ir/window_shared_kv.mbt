///|
/// One request's shared BF16 K/V ring and causal sink attention. Current
/// frame operands remain separate until attention completes, so a prefill
/// chunk larger than the window cannot overwrite history needed by early rows.
pub struct WindowSharedKvPrecision {
  priv heads : Int32
  priv width : Int32
  priv window : Int32
  priv rows : Int32
  priv positions : Int32
  priv sizes : ReadOnlyArray[Int64]
  priv bytes : Int64
  priv probability : AttentionProbabilityLaw
  priv accumulation : AttentionAccumulationLaw
}

///|
pub fn WindowSharedKvPrecision::new(
  query_heads~ : Int32,
  head_width~ : Int32,
  window_tokens~ : Int32,
  maximum_rows~ : Int32,
  maximum_positions~ : Int32,
  maximum_state_bytes~ : Int64,
  probability~ : AttentionProbabilityLaw = GlobalF32,
  accumulation~ : AttentionAccumulationLaw = OrderedFloat32,
) -> Self raise PrecisionPlanError {
  probability.validate()
  guard query_heads > 0 &&
    query_heads <= 1024 &&
    head_width > 0 &&
    head_width <= 65536 &&
    window_tokens > 0 &&
    window_tokens <= maximum_positions &&
    maximum_positions > 0 &&
    maximum_positions <= 1048576 &&
    maximum_rows > 0 &&
    maximum_rows <= maximum_positions else {
    raise IllegalPlacement
  }
  let cells = maximum_rows.to_int64() *
    query_heads.to_int64() *
    head_width.to_int64()
  guard cells <= 2147483647L else { raise ArithmeticOverflow }
  // Window KV words, retained [length,error], append [base,rows,error], output.
  let sizes = [
    window_tokens.to_int64() * head_width.to_int64() * 2L,
    8L,
    12L,
    cells * 2L,
  ]
  let bytes = sizes.fold(init=0L, (total, size) => total + size)
  guard bytes <= maximum_state_bytes else { raise MemoryBudgetExceeded }
  {
    heads: query_heads,
    width: head_width,
    window: window_tokens,
    rows: maximum_rows,
    positions: maximum_positions,
    sizes: ReadOnlyArray::from_array(sizes),
    bytes,
    probability,
    accumulation,
  }
}

///|
pub fn WindowSharedKvPrecision::accumulation_law(
  self : Self,
) -> AttentionAccumulationLaw {
  self.accumulation
}

///|
pub fn WindowSharedKvPrecision::probability_law(
  self : Self,
) -> AttentionProbabilityLaw {
  self.probability
}

///|
pub fn WindowSharedKvPrecision::geometry(
  self : Self,
) -> (Int32, Int32, Int32, Int32, Int32) {
  (self.heads, self.width, self.window, self.rows, self.positions)
}

///|
pub fn WindowSharedKvPrecision::buffer_bytes(
  self : Self,
) -> ReadOnlyArray[Int64] {
  self.sizes
}

///|
pub fn WindowSharedKvPrecision::state_bytes(self : Self) -> Int64 {
  self.bytes
}

///|
pub fn WindowSharedKvPrecision::effect(self : Self) -> NumericEffect {
  ignore(self)
  CacheCommit
}
