///|
pub fn compressed_attention_plan(
  spec : @spec.DeepSeekV4ModelSpec,
  layer~ : Int,
  maximum_rows~ : Int,
  maximum_context~ : Int,
  maximum_state_bytes~ : Int64,
) -> @precision.WindowCompressedKvPrecision raise {
  guard layer >= 0 &&
    layer < spec.layer_count() &&
    maximum_context <= spec.max_position_embeddings() else {
    raise @precision.PrecisionPlanError::IllegalPlacement
  }
  guard spec.compression_ratio(layer) is Some(ratio) && ratio > 0 else {
    raise @precision.PrecisionPlanError::NoImplementation
  }
  @precision.WindowCompressedKvPrecision::new(
    @precision.WindowSharedKvPrecision::new(
      query_heads=spec.attention_head_count(),
      head_width=spec.head_dimension(),
      window_tokens=Int::min(spec.sliding_window(), maximum_context),
      maximum_rows~,
      maximum_positions=maximum_context,
      maximum_state_bytes~,
    ),
    compression_ratio=ratio,
    read_set=if ratio == 4 {
      Learned(top_k=spec.index_top_k())
    } else {
      AllCausal
    },
  )
}

///|
/// Derive aggregate state/scratch from existing immutable component plans.
/// No component may independently consume the complete owner's budget.
fn compressed_cache_sizes(
  spec : @spec.DeepSeekV4ModelSpec,
  layer : Int,
  rows : Int,
  context : Int,
  indexer : Bool,
  state_limit : Int64,
  workspace_limit : Int64,
) -> (Int64, Int64) raise {
  let pool = compressor_pool_plan(
    spec,
    layer~,
    for_indexer=indexer,
    maximum_rows=rows,
    maximum_context=context,
    maximum_state_bytes=state_limit,
    maximum_workspace_bytes=workspace_limit,
  )
  let cache = compressed_cache_plan(
    spec,
    layer~,
    for_indexer=indexer,
    maximum_rows=rows,
    maximum_context=context,
    maximum_state_bytes=state_limit,
    maximum_workspace_bytes=workspace_limit,
  )
  let transform_bytes = if indexer {
    indexer_cache_transform_plan(
      spec,
      layer~,
      maximum_rows=rows,
      maximum_workspace_bytes=workspace_limit,
    ).workspace_bytes()
  } else {
    compressor_cache_plan(
      spec,
      layer~,
      maximum_input_rows=rows,
      maximum_workspace_bytes=workspace_limit,
    ).workspace_bytes()
  }
  (
    pool.state_bytes().fold(init=0L, (sum, size) => sum + size) +
    cache.state_bytes().fold(init=0L, (sum, size) => sum + size),
    pool.frame_bytes().fold(init=0L, (sum, size) => sum + size) +
    cache.workspace_bytes() +
    transform_bytes,
  )
}
