///|
/// Model-neutral rope-free MLA/index projection geometry. Each projection and
/// normalization rounds to BF16; index scoring owns dimension/head scaling.
/// Rotary or quantized index transforms are separate plans, not implicit here.
pub struct IndexedSparseProjectionPrecision {
  priv attention : IndexedSparseAttentionPrecision
  priv hidden : Int
  priv query_rank : Int
  priv kv_rank : Int
  priv rms_epsilon : Double
  priv index_epsilon : Double
  priv frame_sizes : ReadOnlyArray[Int64]
  priv weight_sizes : ReadOnlyArray[Int64]
  priv workspace : Int64
}

///|
pub fn IndexedSparseProjectionPrecision::new(
  attention : IndexedSparseAttentionPrecision,
  hidden_width~ : Int,
  query_rank~ : Int,
  key_value_rank~ : Int,
  rms_epsilon~ : Double,
  index_layer_norm_epsilon~ : Double,
  maximum_workspace_bytes~ : Int64,
) -> Self raise PrecisionPlanError {
  guard hidden_width > 0 &&
    hidden_width <= 65536 &&
    query_rank > 0 &&
    query_rank <= 65536 &&
    key_value_rank > 0 &&
    key_value_rank <= 65536 &&
    rms_epsilon > 0.0 &&
    rms_epsilon <= 1.0 &&
    index_layer_norm_epsilon > 0.0 &&
    index_layer_norm_epsilon <= 1.0 else {
    raise IllegalPlacement
  }
  let (qh, kvh, qkd, vd) = attention.attention_geometry()
  let (ih, id, _, _, _) = attention.index_geometry()
  let (rows, _) = attention.capacity()
  let widths = [
    query_rank,
    query_rank,
    qh * qkd,
    key_value_rank,
    key_value_rank,
    kvh * (qkd + vd),
    kvh * qkd,
    kvh * vd,
    ih * id,
    id,
    id,
    ih,
    id,
  ]
  let sizes = widths.map(width => rows.to_int64() * width.to_int64() * 2L)
  guard sizes.all(size => size <= 4294967294L) else { raise ArithmeticOverflow }
  let workspace = sizes.fold(init=0L, (sum, size) => sum + size)
  guard workspace <= maximum_workspace_bytes else { raise MemoryBudgetExceeded }
  let h = hidden_width.to_int64()
  let q = query_rank.to_int64()
  let k = key_value_rank.to_int64()
  let weights = [
    q * h,
    q,
    qh.to_int64() * qkd.to_int64() * q,
    k * h,
    k,
    kvh.to_int64() * (qkd + vd).to_int64() * k,
    ih.to_int64() * id.to_int64() * q,
    id.to_int64() * h,
    id.to_int64(),
    id.to_int64(),
    ih.to_int64() * h,
    id.to_int64() * h,
  ]
  {
    attention,
    hidden: hidden_width,
    query_rank,
    kv_rank: key_value_rank,
    rms_epsilon,
    index_epsilon: index_layer_norm_epsilon,
    frame_sizes: ReadOnlyArray::from_array(sizes),
    weight_sizes: ReadOnlyArray::from_array(weights.map(size => size * 2L)),
    workspace,
  }
}

///|
pub fn IndexedSparseProjectionPrecision::attention(
  self : Self,
) -> IndexedSparseAttentionPrecision {
  self.attention
}

///|
pub fn IndexedSparseProjectionPrecision::geometry(
  self : Self,
) -> (Int, Int, Int) {
  (self.hidden, self.query_rank, self.kv_rank)
}

///|
pub fn IndexedSparseProjectionPrecision::normalization(
  self : Self,
) -> (Double, Double) {
  (self.rms_epsilon, self.index_epsilon)
}

///|
/// Q-low, Q-normalized, Q, KV-low, KV-normalized, packed-KV, K, V,
/// index-Q, raw-index-K, index-K, index-head-weights, pool-gates.
pub fn IndexedSparseProjectionPrecision::frame_bytes(
  self : Self,
) -> ReadOnlyArray[Int64] {
  self.frame_sizes
}

///|
/// Q-A, Q-norm, Q-B, KV-A, KV-norm, KV-B, index-Q, index-K,
/// index-norm scale/bias, index-head-weight projection, pool-gate projection.
pub fn IndexedSparseProjectionPrecision::weight_bytes(
  self : Self,
) -> ReadOnlyArray[Int64] {
  self.weight_sizes
}

///|
pub fn IndexedSparseProjectionPrecision::workspace_bytes(self : Self) -> Int64 {
  self.workspace
}
