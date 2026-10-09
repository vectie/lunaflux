///|
/// Pure projected sparse attention geometry. Query frames and retained key
/// history have independent capacities. Pooling and stable index selection
/// precede GQA attention; all intermediate sizes are computed before allocation.
pub struct IndexedSparseAttentionPrecision {
  priv index_heads : Int
  priv index_dimension : Int
  priv query_heads : Int
  priv kv_heads : Int
  priv qk_dimension : Int
  priv value_dimension : Int
  priv pool_size : Int
  priv top_k : Int
  priv tail : Bool
  priv queries : Int
  priv history : Int
  priv frame_sizes : ReadOnlyArray[Int64]
  priv workspace : Int64
}

///|
pub fn IndexedSparseAttentionPrecision::new(
  index_head_count~ : Int,
  index_head_dimension~ : Int,
  query_head_count~ : Int,
  key_value_head_count~ : Int,
  query_key_dimension~ : Int,
  value_dimension~ : Int,
  pool_size~ : Int,
  top_k~ : Int,
  append_incomplete_tail~ : Bool,
  maximum_queries~ : Int,
  maximum_history~ : Int,
  maximum_workspace_bytes~ : Int64,
) -> Self raise PrecisionPlanError {
  guard index_head_count > 0 &&
    index_head_count <= 1024 &&
    index_head_dimension > 0 &&
    index_head_dimension <= 65536 &&
    query_head_count > 0 &&
    query_head_count <= 1024 &&
    key_value_head_count > 0 &&
    key_value_head_count <= query_head_count &&
    query_head_count % key_value_head_count == 0 &&
    query_key_dimension > 0 &&
    query_key_dimension <= 65536 &&
    value_dimension > 0 &&
    value_dimension <= 65536 &&
    pool_size > 0 &&
    pool_size <= 1024 &&
    top_k > 0 &&
    top_k <= 65536 &&
    top_k % pool_size == 0 &&
    maximum_history >= pool_size &&
    maximum_history <= 1048576 &&
    maximum_queries > 0 &&
    maximum_queries <= maximum_history else {
    raise IllegalPlacement
  }
  let pools = (maximum_history / pool_size).to_int64()
  let width = top_k + (if append_incomplete_tail { pool_size - 1 } else { 0 })
  guard width <= 65536 else { raise IllegalPlacement }
  let selected_cells = expert_checked_mul(
    maximum_queries.to_int64(),
    width.to_int64(),
  )
  let output_cells = expert_checked_mul(
    expert_checked_mul(maximum_queries.to_int64(), query_head_count.to_int64()),
    value_dimension.to_int64(),
  )
  guard selected_cells <= 2147483647L && output_cells <= 2147483647L else {
    raise ArithmeticOverflow
  }
  let sizes = [
    expert_checked_mul(
      expert_checked_mul(pools, index_head_dimension.to_int64()),
      2L,
    ),
    expert_checked_mul(expert_checked_mul(pools, pool_size.to_int64()), 4L),
    expert_checked_mul(pools, 4L),
    4L,
    expert_checked_mul(selected_cells, 4L),
  ]
  // Five bounded sizes: history <= 2^20, dimensions <= 2^16 and selected
  // cells <= 2^31-1. Their aggregate cannot overflow Int64.
  let workspace = sizes.fold(init=0L, (sum, size) => sum + size)
  guard workspace <= maximum_workspace_bytes else { raise MemoryBudgetExceeded }
  {
    index_heads: index_head_count,
    index_dimension: index_head_dimension,
    query_heads: query_head_count,
    kv_heads: key_value_head_count,
    qk_dimension: query_key_dimension,
    value_dimension,
    pool_size,
    top_k,
    tail: append_incomplete_tail,
    queries: maximum_queries,
    history: maximum_history,
    frame_sizes: ReadOnlyArray::from_array(sizes),
    workspace,
  }
}

///|
pub fn IndexedSparseAttentionPrecision::index_geometry(
  self : Self,
) -> (Int, Int, Int, Int, Bool) {
  (
    self.index_heads,
    self.index_dimension,
    self.pool_size,
    self.top_k,
    self.tail,
  )
}

///|
pub fn IndexedSparseAttentionPrecision::attention_geometry(
  self : Self,
) -> (Int, Int, Int, Int) {
  (self.query_heads, self.kv_heads, self.qk_dimension, self.value_dimension)
}

///|
pub fn IndexedSparseAttentionPrecision::capacity(self : Self) -> (Int, Int) {
  (self.queries, self.history)
}

///|
pub fn IndexedSparseAttentionPrecision::selected_width(self : Self) -> Int {
  self.top_k + (if self.tail { self.pool_size - 1 } else { 0 })
}

///|
pub fn IndexedSparseAttentionPrecision::frame_bytes(
  self : Self,
) -> ReadOnlyArray[Int64] {
  self.frame_sizes
}

///|
pub fn IndexedSparseAttentionPrecision::workspace_bytes(self : Self) -> Int64 {
  self.workspace
}
