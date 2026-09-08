///|
/// Immutable block tiling and lifetime plan for an ordered matrix fold.
/// Two transfer slots alternate; reduction updates retain their order.
pub struct ProjectionFoldPipeline {
  priv tile_width : Int
  priv tile_count : Int
  priv transfer_width : Int
} derive(Debug, Eq)

///|
pub fn ProjectionFoldPipeline::tile_width(self : ProjectionFoldPipeline) -> Int {
  self.tile_width
}

///|
pub fn ProjectionFoldPipeline::tile_count(self : ProjectionFoldPipeline) -> Int {
  self.tile_count
}

///|
pub fn ProjectionFoldPipeline::stage_count(
  _self : ProjectionFoldPipeline,
) -> Int {
  2
}

///|
pub fn ProjectionFoldPipeline::transfer_width(
  self : ProjectionFoldPipeline,
) -> Int {
  self.transfer_width
}

///|
pub fn ProjectionFoldPipeline::row_width(_self : ProjectionFoldPipeline) -> Int {
  64
}

///|
pub fn ProjectionFoldPipeline::single_row_outputs(
  _self : ProjectionFoldPipeline,
) -> Int {
  4
}

///|
pub fn ProjectionFoldPipeline::minimum_rows(
  self : ProjectionFoldPipeline,
) -> Int {
  self.row_width() * 4
}

///|
fn projection_intermediate_pipeline(
  optimized : OptimizedProjectionTileCompilation,
) -> ProjectionFoldPipeline? {
  let dispatch = optimized.semantic.selected.request.dispatch
  let (_, _, extent) = dispatch.shape().dimensions()
  let (_, _, tile) = dispatch.prefill().tile().dimensions()
  // This closed semantic graph has a completed materialization boundary:
  // operand reads are immutable and independent of the accumulator. Do not
  // move reads across arbitrary stores, KV commits, or other effectful folds.
  if optimized.optimization.applied(ReuseIntermediateTileAcrossOutputTiles) &&
    dispatch.prefill().kernel_class() == MatrixTile &&
    extent >= 512 &&
    tile > 0 &&
    extent % (tile * 4) == 0 {
    Some({
      tile_width: tile,
      tile_count: extent / tile,
      transfer_width: tile * 2,
    })
  } else {
    None
  }
}

///|
pub fn ProjectionTileSchedule::intermediate_pipeline(
  self : ProjectionTileSchedule,
) -> ProjectionFoldPipeline? {
  self.intermediate_pipeline
}
