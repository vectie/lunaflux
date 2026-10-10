///|
/// Startup-selected phases. Priming retains main KV without evaluating draft
/// attention. Prediction publishes one committed row before noncausal draft reads.
pub(all) enum CommittedDraftPhase {
  Prime
  Predict
  PredictCommitted
} derive(Eq, Debug)

///|
pub extend CommittedDraftPhase with Eq::{equal, not_equal}

///|
pub extend CommittedDraftPhase with @debug.Debug::{to_repr}

///|
/// Reading an already-published prefix must not republish or advance it.
pub fn CommittedDraftPhase::effect(self : Self) -> NumericEffect {
  match self {
    PredictCommitted => StepWrite
    Prime | Predict => CacheCommit
  }
}

///|
/// Immutable arithmetic/read-set plan. Main and draft ports are independent;
/// draft values never enter the committed ring. Physical slot order is part of
/// the numerical contract, not a backend-specific scheduling choice.
pub struct CommittedDraftKvPrecision {
  priv heads : Int
  priv width : Int
  priv window : Int
  priv main_rows : Int
  priv draft_rows : Int
  priv positions : Int
  priv sizes : ReadOnlyArray[Int64]
  priv bytes : Int64
}

///|
pub fn CommittedDraftKvPrecision::new(
  query_heads~ : Int,
  head_width~ : Int,
  window_tokens~ : Int,
  maximum_main_rows~ : Int,
  maximum_draft_rows~ : Int,
  maximum_positions~ : Int,
  maximum_state_bytes~ : Int64,
) -> Self raise PrecisionPlanError {
  guard query_heads > 0 &&
    query_heads <= 1024 &&
    head_width > 0 &&
    head_width <= 65536 &&
    window_tokens > 0 &&
    window_tokens <= maximum_positions &&
    maximum_main_rows > 0 &&
    maximum_main_rows <= maximum_positions &&
    maximum_draft_rows > 0 &&
    maximum_draft_rows <= maximum_positions &&
    maximum_positions > 0 &&
    maximum_positions <= 1048576 else {
    raise IllegalPlacement
  }
  let cells = maximum_draft_rows.to_int64() *
    query_heads.to_int64() *
    head_width.to_int64()
  let main_cells = maximum_main_rows.to_int64() * head_width.to_int64()
  guard cells <= 2147483647L && main_cells <= 2147483647L else {
    raise ArithmeticOverflow
  }
  // Ring, [committed frontier,error], [base,main,draft,error], BF16 output.
  let sizes = [
    window_tokens.to_int64() * head_width.to_int64() * 2L,
    8L,
    16L,
    cells * 2L,
  ]
  let bytes = sizes.fold(init=0L, (total, size) => total + size)
  guard bytes <= maximum_state_bytes else { raise MemoryBudgetExceeded }
  {
    heads: query_heads,
    width: head_width,
    window: window_tokens,
    main_rows: maximum_main_rows,
    draft_rows: maximum_draft_rows,
    positions: maximum_positions,
    sizes: ReadOnlyArray::from_array(sizes),
    bytes,
  }
}

///|
pub fn CommittedDraftKvPrecision::geometry(
  self : Self,
) -> (Int, Int, Int, Int, Int, Int) {
  (
    self.heads,
    self.width,
    self.window,
    self.main_rows,
    self.draft_rows,
    self.positions,
  )
}

///|
pub fn CommittedDraftKvPrecision::buffer_bytes(
  self : Self,
) -> ReadOnlyArray[Int64] {
  self.sizes
}

///|
pub fn CommittedDraftKvPrecision::state_bytes(self : Self) -> Int64 {
  self.bytes
}

///|
pub fn CommittedDraftKvPrecision::effect(self : Self) -> NumericEffect {
  ignore(self)
  CacheCommit
}

///|
/// Every draft row reads these physical slots, then every current draft KV
/// row. In particular, the read set is independent of the query's draft index.
pub fn CommittedDraftKvPrecision::visible_main_slots(
  self : Self,
  committed_position : Int,
) -> Int raise PrecisionPlanError {
  guard committed_position > 0 && committed_position < self.positions else {
    raise IllegalPlacement
  }
  Int::min(self.window, committed_position + 1)
}
