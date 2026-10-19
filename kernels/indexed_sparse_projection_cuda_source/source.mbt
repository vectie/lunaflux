///|
/// AOT lowering of the rope-free projected sparse chain. This composes existing
/// ordered projections and RMSNorm, plus explicit affine LayerNorm for index K.
pub struct IndexedSparseProjectionCudaSource {
  priv plan : @precision.IndexedSparseProjectionPrecision
  priv symbols : ReadOnlyArray[String]
  priv sources : Bytes
  priv geometry : ReadOnlyArray[(Int, Int)]
}

///|
pub fn indexed_sparse_projection_source(
  plan : @precision.IndexedSparseProjectionPrecision,
  symbol_prefix~ : String,
  staging? : @matrix.MatrixStaging = Synchronous,
) -> IndexedSparseProjectionCudaSource raise {
  guard staging == Synchronous || plan.projection_law() != ScalarFloat32 else {
    raise @precision.PrecisionPlanError::NoImplementation
  }
  let (hidden, qr, kr) = plan.geometry()
  let (qh, kvh, qkd, vd) = plan.attention().attention_geometry()
  let (ih, id, _, _, _) = plan.attention().index_geometry()
  let (rows, _) = plan.attention().capacity()
  let (rms_eps, index_eps) = plan.normalization()
  let names = [
    "q_a", "q_norm", "q_b", "kv_a", "kv_norm", "kv_b", "kv_split", "index_q", "index_k",
    "index_norm", "index_weights", "pool_gate",
  ]
  if plan.cache_law() is LatentBFloat16 {
    names.push("q_absorb")
  }
  let symbols = names.map(name => symbol_prefix + "_" + name)
  let dense = [
    (hidden, qr, 0),
    (qr, qh * qkd, 2),
    (hidden, kr, 3),
    (kr, kvh * (qkd + vd), 5),
    (qr, ih * id, 7),
    (hidden, id, 8),
    (hidden, ih, 10),
    (hidden, id, 11),
  ]
  let sources : Array[Bytes] = Array::make(names.length(), b"")
  let launches : Array[(Int, Int)] = Array::make(
    names.length(),
    (rows, 256),
  )
  for (input, output, stage) in dense {
    if stage == 5 && plan.cache_law() is LatentBFloat16 {
      continue
    }
    if plan.projection_law() != ScalarFloat32 {
      let tile = @matrix.MatrixTilePlan::new(
        rows~,
        columns=output,
        inner=input,
        law=plan.projection_law(),
        staging~,
      )
      sources[stage] = @tiled.dense_source(tile, symbol=symbols[stage])
      launches[stage] = (tile.tile_count(), 128)
      continue
    }
    sources[stage] = @projection.dense_projection_bf16_source(
      input_width=input,
      output_width=output,
      maximum_rows=rows,
      function_symbol=symbols[stage],
    ).source_bytes()
    launches[stage] = (
      ((rows.to_int64() * output.to_int64() + 255L) / 256L).to_int(),
      256,
    )
  }
  for (width, stage) in [(qr, 1), (kr, 4)] {
    sources[stage] = @rms.prefix_rms_norm_bf16_with_optional_suffix_copy_source(
      prefix_width=width,
      suffix_width=0,
      maximum_rows=rows,
      epsilon=rms_eps,
      function_symbol=symbols[stage],
    ).source_bytes()
  }
  sources[6] = @kv.decoder_kv_assembly_bf16_source(
    head_count=kvh,
    key_nope_width=qkd,
    shared_key_suffix_width=0,
    value_width=vd,
    maximum_rows=rows,
    function_symbol=symbols[6],
  ).source_bytes()
  launches[6] = (
    ((rows.to_int64() * kvh.to_int64() * (qkd + vd).to_int64() + 255L) / 256L).to_int(),
    256,
  )
  let norm = @norm.affine_layer_norm_bf16_source(
    width=id,
    maximum_rows=rows,
    epsilon=index_eps,
    function_symbol=symbols[9],
  )
  sources[9] = norm.source_bytes()
  launches[9] = norm.launch()
  if plan.cache_law() is LatentBFloat16 {
    sources[6] = b""
    let absorb = latent_head_plan(plan, false)
    sources[12] = @tiled.head_projection_source(absorb, symbol=symbols[12])
    launches[12] = (absorb.workgroups(), 128)
  }
  {
    plan,
    symbols: ReadOnlyArray::from_array(symbols),
    sources: sources.fold(init=b"", (result, source) => result + source + b"\n"),
    geometry: ReadOnlyArray::from_array(launches),
  }
}

///|
pub fn IndexedSparseProjectionCudaSource::plan(
  self : Self,
) -> @precision.IndexedSparseProjectionPrecision {
  self.plan
}

///|
pub fn IndexedSparseProjectionCudaSource::source_bytes(self : Self) -> Bytes {
  self.sources
}

///|
pub fn IndexedSparseProjectionCudaSource::symbol(
  self : Self,
  stage : Int,
) -> String {
  self.symbols[stage]
}

///|
pub fn IndexedSparseProjectionCudaSource::launch(
  self : Self,
  stage : Int,
) -> (Int, Int) {
  self.geometry[stage]
}
