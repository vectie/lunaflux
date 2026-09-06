///|
/// Pure progress-to-work calculation. Capacity is a logical token bound, not
/// a device allocation. Zero means no runnable work under the supplied budget.
pub fn prefill_query_count(
  computed~ : Int,
  target~ : Int,
  token_budget~ : Int,
  chunk_limit~ : Int,
  kv_token_capacity~ : Int,
) -> Int {
  if computed < 0 ||
    target <= computed ||
    token_budget <= 0 ||
    chunk_limit <= 0 ||
    kv_token_capacity <= computed {
    return 0
  }
  Int::min(
    Int::min(target - computed, kv_token_capacity - computed),
    Int::min(token_budget, chunk_limit),
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
  priv packed_start : Int
  priv count : Int
  priv computed : Int
  priv context_end : Int
  priv output_demand : Bool
} derive(Debug, Eq)

///|
pub fn RowWork::new(
  packed_start~ : Int,
  computed~ : Int,
  query_count~ : Int,
  kv_token_capacity~ : Int,
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
pub fn RowWork::query_start(self : RowWork) -> Int {
  self.packed_start
}

///|
pub fn RowWork::query_end(self : RowWork) -> Int {
  self.packed_start + self.count
}

///|
pub fn RowWork::query_count(self : RowWork) -> Int {
  self.count
}

///|
pub fn RowWork::cached_context_end(self : RowWork) -> Int {
  self.computed
}

///|
pub fn RowWork::context_end(self : RowWork) -> Int {
  self.context_end
}

///|
pub fn RowWork::kv_write_start(self : RowWork) -> Int {
  self.computed
}

///|
pub fn RowWork::kv_write_end(self : RowWork) -> Int {
  self.context_end
}

///|
/// -1 means no externally observed output row. It is a host planning value,
/// not an overloaded GPU descriptor offset.
pub fn RowWork::output_token_row(self : RowWork) -> Int {
  if self.output_demand {
    self.query_end() - 1
  } else {
    -1
  }
}
