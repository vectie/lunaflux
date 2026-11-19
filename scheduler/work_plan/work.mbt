///|
/// An aged prompt retains one row and one token of forward progress; ready
/// decodes retain one token each within the remaining row/token domain.
/// This is backend-neutral fairness planning, not a device tile heuristic.
pub fn aged_prefill_token_budget(
  token_budget~ : Int32,
  row_limit~ : Int32,
  runnable_decodes~ : Int32,
) -> Int32 {
  if token_budget <= 0 || row_limit <= 0 {
    return 0
  }
  let reserved = Int32::min(
    Int32::max(runnable_decodes, 0),
    Int32::min(row_limit - 1, token_budget - 1),
  )
  token_budget - reserved
}

///|
/// Pure progress-to-work calculation. Capacity is a logical token bound, not
/// a device allocation. Zero means no runnable work under the supplied budget.
pub fn prefill_query_count(
  computed~ : Int32,
  target~ : Int32,
  token_budget~ : Int32,
  chunk_limit~ : Int32,
  kv_token_capacity~ : Int32,
) -> Int32 {
  if computed < 0 ||
    target <= computed ||
    token_budget <= 0 ||
    chunk_limit <= 0 ||
    kv_token_capacity <= computed {
    return 0
  }
  Int32::min(
    Int32::min(target - computed, kv_token_capacity - computed),
    Int32::min(token_budget, chunk_limit),
  )
}

///|
pub suberror WorkPlanError {
  InvalidRange
} derive(Debug, Eq)

///|
/// Half-open ranges for one logical row. Pure value semantics; no arrays,
/// ownership mutation or backend identity. KV writes survive absent logits.
#valtype
pub struct RowWork {
  priv packed_start : Int32
  priv count : Int32
  priv computed : Int32
  priv context_end : Int32
  priv output_demand : Bool
} derive(Debug, Eq)

///|
pub fn RowWork::new(
  packed_start~ : Int32,
  computed~ : Int32,
  query_count~ : Int32,
  kv_token_capacity~ : Int32,
  output_demand~ : Bool,
) -> RowWork raise WorkPlanError {
  if packed_start < 0 ||
    computed < 0 ||
    query_count <= 0 ||
    packed_start > 2147483647 - query_count ||
    kv_token_capacity < query_count ||
    computed > kv_token_capacity - query_count {
    raise InvalidRange
  }
  {
    packed_start,
    computed,
    count: query_count,
    context_end: computed + query_count,
    output_demand,
  }
}

///|
pub fn RowWork::query_start(self : RowWork) -> Int32 {
  self.packed_start
}

///|
pub fn RowWork::query_end(self : RowWork) -> Int32 {
  self.packed_start + self.count
}

///|
pub fn RowWork::query_count(self : RowWork) -> Int32 {
  self.count
}

///|
pub fn RowWork::cached_context_end(self : RowWork) -> Int32 {
  self.computed
}

///|
pub fn RowWork::context_end(self : RowWork) -> Int32 {
  self.context_end
}

///|
pub fn RowWork::kv_write_start(self : RowWork) -> Int32 {
  self.computed
}

///|
pub fn RowWork::kv_write_end(self : RowWork) -> Int32 {
  self.context_end
}

///|
/// -1 means no externally observed output row. It is a host planning value,
/// not an overloaded GPU descriptor offset.
pub fn RowWork::output_token_row(self : RowWork) -> Int32 {
  if self.output_demand {
    self.query_end() - 1
  } else {
    -1
  }
}

///|
pub extend WorkPlanError with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend WorkPlanError with Eq::{not_equal, equal}

///|
pub extend RowWork with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend RowWork with Eq::{not_equal, equal}
