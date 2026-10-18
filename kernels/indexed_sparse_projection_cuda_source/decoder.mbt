///|
/// CUDA lowering only; the immutable composition above owns geometry/budgets.
pub struct IndexedSparseDecoderCudaSource {
  priv plan : @precision.IndexedSparseDecoderPrecision
  priv projection : IndexedSparseProjectionCudaSource
  priv history : @sparse.IndexedSparseHistoryCudaSource
  priv attention : @sparse.IndexedSparseAttentionCudaSource
  priv envelope : @hyper.HyperConnectionCudaSource
  priv output : Bytes
  priv output_symbol : String
  priv output_launch : (Int, Int)
}

///|
pub fn indexed_sparse_decoder_source(
  plan : @precision.IndexedSparseDecoderPrecision,
  symbol_prefix~ : String,
  staging? : @matrix.MatrixStaging = Synchronous,
) -> IndexedSparseDecoderCudaSource raise {
  let projection = plan.projection()
  let attention = projection.attention()
  let (heads, _, _, values) = attention.attention_geometry()
  let (hidden, _, _) = projection.geometry()
  let (rows, _) = attention.capacity()
  let output_symbol = symbol_prefix + "_output"
  let partial = plan.publication() == ReduceFloat32BeforeRound
  let (output, output_launch) = if projection.projection_law() != ScalarFloat32 {
    let tile = @matrix.MatrixTilePlan::new(
      rows~,
      columns=hidden,
      inner=heads * values,
      law=projection.projection_law(),
      staging~,
    )
    (
      @tiled.dense_source(tile, symbol=output_symbol, partial_float32=partial),
      (tile.tile_count(), 128),
    )
  } else if partial {
    (
      @tiled.scalar_partial_source(
        rows~,
        columns=hidden,
        inner=heads * values,
        symbol=output_symbol,
      ),
      (((rows.to_int64() * hidden.to_int64() + 255L) / 256L).to_int(), 256),
    )
  } else {
    (
      @projection.dense_projection_bf16_source(
        input_width=heads * values,
        output_width=hidden,
        maximum_rows=rows,
        function_symbol=output_symbol,
      ).source_bytes(),
      (((rows.to_int64() * hidden.to_int64() + 255L) / 256L).to_int(), 256),
    )
  }
  {
    plan,
    projection: indexed_sparse_projection_source(
      projection,
      symbol_prefix=symbol_prefix + "_projection",
      staging~,
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
    output: output +
    (if partial {
      b"\n" +
      @tiled.publication_source(
        rows~,
        width=hidden,
        symbol=output_symbol + "_publish",
      )
    } else {
      b""
    }),
    output_symbol,
    output_launch,
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
  self.output_symbol
}

///|
pub fn IndexedSparseDecoderCudaSource::output_launch(self : Self) -> (Int, Int) {
  self.output_launch
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
  self.output
}
