///|
/// Immutable block tiling and lifetime plan for an ordered matrix fold.
/// Two transfer slots alternate; reduction updates retain their order.
pub struct ProjectionFoldPipeline {
  priv tile_width : Int
  priv tile_count : Int
  priv transfer_width : Int
  priv row_width : Int
  priv output_tiles : Int
  priv minimum_rows : Int
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
pub fn ProjectionFoldPipeline::row_width(self : ProjectionFoldPipeline) -> Int {
  self.row_width
}

///|
pub fn ProjectionFoldPipeline::output_tiles(
  self : ProjectionFoldPipeline,
) -> Int {
  self.output_tiles
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
  self.minimum_rows
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
      row_width: 64,
      minimum_rows: 256,
      output_tiles: projection_companion_output_tiles(
        dispatch.prefill().output_tiles_per_workgroup(),
      ),
    })
  } else {
    None
  }
}

///|
fn projection_sibling_pipeline(
  optimized : OptimizedProjectionTileCompilation,
) -> ProjectionFoldPipeline? {
  let dispatch = optimized.semantic.selected.request.dispatch
  let (extent, _, _) = dispatch.shape().dimensions()
  let (_, _, tile) = dispatch.prefill().tile().dimensions()
  if optimized.optimization.applied(FuseSiblingInputTraversals) &&
    dispatch.prefill().kernel_class() == MatrixTile &&
    extent >= 512 &&
    tile > 0 &&
    extent % (tile * 4) == 0 {
    Some({
      tile_width: tile,
      tile_count: extent / tile,
      transfer_width: tile * 2,
      row_width: 64,
      minimum_rows: 256,
      output_tiles: Int::max(
        1,
        dispatch.prefill().output_tiles_per_workgroup() / 4,
      ),
    })
  } else {
    None
  }
}

///|
pub fn ProjectionTileSchedule::sibling_pipeline(
  self : ProjectionTileSchedule,
) -> ProjectionFoldPipeline? {
  self.sibling_pipeline
}

///|
pub fn ProjectionTileSchedule::intermediate_pipeline(
  self : ProjectionTileSchedule,
) -> ProjectionFoldPipeline? {
  self.intermediate_pipeline
}

///|
fn projection_matrix_pipeline(
  optimized : OptimizedProjectionTileCompilation,
) -> ProjectionFoldPipeline? {
  let request = optimized.semantic.selected.request
  let dispatch = request.dispatch
  let (extent, _, _) = dispatch.shape().dimensions()
  let (m, n, k) = dispatch.prefill().tile().dimensions()
  let selected = request.demand == QueryRowEnds
  if dispatch.shape().family() == GatedMlp ||
    optimized.optimization.applied(FuseAttentionIngressEpilogue) ||
    (selected && dispatch.prefill().output_tiles_per_workgroup() == 1) ||
    dispatch.prefill().kernel_class() != MatrixTile ||
    (m, n, k) != (16, 16, 16) ||
    extent < 512 ||
    extent % 64 != 0 ||
    dispatch.prefill().input_residency() == SelectedRowTileResidency {
    None
  } else {
    Some({
      tile_width: k,
      tile_count: extent / k,
      // A bounded selected-row product can amortize transfers over four
      // ordered microtiles. Wider output products retain the smaller slot.
      transfer_width: if selected &&
        dispatch.prefill().output_tiles_per_workgroup() > 8 {
        k * 2
      } else {
        k * 4
      },
      row_width: if selected {
        16
      } else {
        32
      },
      output_tiles: if selected {
        dispatch.prefill().output_tiles_per_workgroup()
      } else {
        Int::max(1, dispatch.prefill().output_tiles_per_workgroup() / 2)
      },
      minimum_rows: if selected {
        2
      } else {
        256
      },
    })
  }
}

///|
/// The matrix pipeline preserves each ordered dot fold while sharing immutable
/// operands across independent rows. Sparse demand is retained by the schedule.
pub fn ProjectionTileSchedule::matrix_pipeline(
  self : ProjectionTileSchedule,
) -> ProjectionFoldPipeline? {
  self.matrix_pipeline
}
