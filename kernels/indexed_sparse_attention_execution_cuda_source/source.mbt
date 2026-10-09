///|
/// CUDA lowering of a generic projected sparse attention plan. The arithmetic
/// remains the existing ordered BF16/F32 law; these reference schedules are not
/// throughput-optimized serving kernels.
pub struct IndexedSparseAttentionCudaSource {
  priv plan : @precision.IndexedSparseAttentionPrecision
  priv symbols : ReadOnlyArray[String]
  priv source : Bytes
}

///|
pub fn indexed_sparse_attention_source(
  plan : @precision.IndexedSparseAttentionPrecision,
  symbol_prefix~ : String,
) -> IndexedSparseAttentionCudaSource raise {
  let (ih, id, pool_size, top_k, tail) = plan.index_geometry()
  let (qh, kvh, qkd, vd) = plan.attention_geometry()
  let (queries, history) = plan.capacity()
  let symbols = [
    symbol_prefix + "_pool",
    symbol_prefix + "_index",
    symbol_prefix + "_attention",
  ]
  let pool = @pooling.dsa_key_pool_compression_bf16_source(
    head_dimension=id,
    pool_size~,
    maximum_tokens=history,
    function_symbol=symbols[0],
  )
  let index = @index.projected_index_score_stable_topk_bf16_source(
    head_count=ih,
    head_dimension=id,
    top_k~,
    pool_size~,
    append_incomplete_tail=tail,
    maximum_tokens=history,
    maximum_queries=queries,
    function_symbol=symbols[1],
  )
  let attention = @attention.selected_index_sparse_attention_bf16_source(
    query_head_count=qh,
    key_value_head_count=kvh,
    query_key_dimension=qkd,
    value_dimension=vd,
    selected_width=plan.selected_width(),
    maximum_tokens=history,
    maximum_queries=queries,
    function_symbol=symbols[2],
  )
  {
    plan,
    symbols: ReadOnlyArray::from_array(symbols),
    source: pool.source_bytes() +
    b"\n" +
    index.source_bytes() +
    b"\n" +
    attention.source_bytes(),
  }
}

///|
pub fn IndexedSparseAttentionCudaSource::plan(
  self : Self,
) -> @precision.IndexedSparseAttentionPrecision {
  self.plan
}

///|
pub fn IndexedSparseAttentionCudaSource::symbol(
  self : Self,
  stage : Int,
) -> String {
  self.symbols[stage]
}

///|
pub fn IndexedSparseAttentionCudaSource::source_bytes(self : Self) -> Bytes {
  self.source
}
