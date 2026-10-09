///|
/// CUDA lowering only; the immutable composition above owns geometry/budgets.
pub struct IndexedSparseDecoderCudaSource {
  priv plan : @precision.IndexedSparseDecoderPrecision
  priv projection : IndexedSparseProjectionCudaSource
  priv history : @sparse.IndexedSparseHistoryCudaSource
  priv attention : @sparse.IndexedSparseAttentionCudaSource
  priv envelope : @hyper.HyperConnectionCudaSource
  priv output : @projection.DecoderProjectionCudaSource
}

///|
pub fn indexed_sparse_decoder_source(
  plan : @precision.IndexedSparseDecoderPrecision,
  symbol_prefix~ : String,
) -> IndexedSparseDecoderCudaSource raise {
  let projection = plan.projection()
  let attention = projection.attention()
  let (heads, _, _, values) = attention.attention_geometry()
  let (hidden, _, _) = projection.geometry()
  let (rows, _) = attention.capacity()
  {
    plan,
    projection: indexed_sparse_projection_source(
      projection,
      symbol_prefix=symbol_prefix + "_projection",
    ),
    history: @sparse.indexed_sparse_history_source(
      plan.history(),
      symbol_prefix=symbol_prefix + "_history",
    ),
    attention: @sparse.indexed_sparse_attention_source(
      attention,
      symbol_prefix=symbol_prefix + "_attention",
    ),
    envelope: @hyper.hyper_connection_source(
      plan.envelope(),
      symbol_prefix=symbol_prefix + "_hyper",
    ),
    output: @projection.dense_projection_bf16_source(
      input_width=heads * values,
      output_width=hidden,
      maximum_rows=rows,
      function_symbol=symbol_prefix + "_output",
    ),
  }
}

///|
pub fn IndexedSparseDecoderCudaSource::plan(
  self : Self,
) -> @precision.IndexedSparseDecoderPrecision {
  self.plan
}

///|
pub fn IndexedSparseDecoderCudaSource::projection(
  self : Self,
) -> IndexedSparseProjectionCudaSource {
  self.projection
}

///|
pub fn IndexedSparseDecoderCudaSource::history(
  self : Self,
) -> @sparse.IndexedSparseHistoryCudaSource {
  self.history
}

///|
pub fn IndexedSparseDecoderCudaSource::attention(
  self : Self,
) -> @sparse.IndexedSparseAttentionCudaSource {
  self.attention
}

///|
pub fn IndexedSparseDecoderCudaSource::envelope(
  self : Self,
) -> @hyper.HyperConnectionCudaSource {
  self.envelope
}

///|
pub fn IndexedSparseDecoderCudaSource::output_symbol(self : Self) -> String {
  self.output.function_symbol()
}

///|
pub fn IndexedSparseDecoderCudaSource::source_bytes(self : Self) -> Bytes {
  self.projection.source_bytes() +
  b"\n" +
  self.history.source_bytes() +
  b"\n" +
  self.attention.source_bytes() +
  b"\n" +
  self.envelope.source_bytes() +
  b"\n" +
  self.output.source_bytes()
}
