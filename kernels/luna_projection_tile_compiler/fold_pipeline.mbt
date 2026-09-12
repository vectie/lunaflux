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
  let request = optimized.semantic.selected.request
  let dispatch = request.dispatch
  let (_, _, extent) = dispatch.shape().dimensions()
  let (_, _, tile) = dispatch.prefill().tile().dimensions()
  // This closed semantic graph has a completed materialization boundary:
  // operand reads are immutable and independent of the accumulator. Do not
  // move reads across arbitrary stores, KV commits, or other effectful folds.
  if optimized.optimization.applied(ReuseIntermediateTileAcrossOutputTiles) &&
    dispatch.prefill().kernel_class() == MatrixTile &&
    extent >= 256 &&
    tile > 0 &&
    extent % (tile * 4) == 0 {
    // Specialize the independent output-row map to this executable's demand.
    // The reduction tiles and their order are unchanged; an eight-row bucket
    // needs one padded matrix row tile, not four fully evaluated row tiles.
    guard projection_bounded_row_extent(
        maximum_rows=request.maximum_tokens,
        matrix_rows=16,
        preferred_rows=32,
      )
      is Some(row_width) else {
      return None
    }
    Some({
      tile_width: tile,
      tile_count: extent / tile,
      // Long folds with a narrow independent product can amortize producer
      // waits without making the live operand footprint dominate residency.
      transfer_width: projection_transfer_width(
        extent,
        tile,
        projection_operand_window_tiles(
          extent,
          row_width,
          projection_companion_output_tiles(
            dispatch.prefill().output_tiles_per_workgroup(),
          ),
          maximum_tiles=8,
        ),
      ),
      row_width,
      minimum_rows: 2,
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
  let request = optimized.semantic.selected.request
  let dispatch = request.dispatch
  let (extent, _, _) = dispatch.shape().dimensions()
  let (matrix_rows, _, tile) = dispatch.prefill().tile().dimensions()
  if optimized.optimization.applied(FuseSiblingInputTraversals) &&
    dispatch.prefill().kernel_class() == MatrixTile &&
    extent >= 256 &&
    tile > 0 &&
    extent % (tile * 4) == 0 {
    // Restrict the independent row product before storage and consumer
    // partitioning. The ordered reduction and actual-row masks are intact.
    guard projection_bounded_row_extent(
        maximum_rows=request.maximum_tokens,
        matrix_rows~,
        preferred_rows=64,
      )
      is Some(row_width) else {
      return None
    }
    Some({
      tile_width: tile,
      tile_count: extent / tile,
      // A bounded single-row-tile product has room for four immutable fold
      // fragments per transfer. Wider row products retain the smaller live
      // operand window. Reduction updates still consume one tile at a time.
      transfer_width: tile *
      (if row_width == matrix_rows &&
        dispatch.prefill().output_tiles_per_workgroup() <= 8 {
        4
      } else {
        2
      }),
      row_width,
      minimum_rows: 2,
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
    extent <= 0 ||
    extent % k != 0 ||
    dispatch.prefill().input_residency() == SelectedRowTileResidency {
    None
  } else {
    Some({
      tile_width: k,
      tile_count: extent / k,
      // Bound lookahead by both fold length and the independent product.
      // Short folds and wide products lose more residency than they gain in
      // overlap. Selected gathers retain their separately measured window.
      transfer_width: projection_transfer_width(
        extent,
        k,
        if selected && dispatch.prefill().output_tiles_per_workgroup() > 8 {
          2
        } else if !selected {
          projection_operand_window_tiles(
            extent,
            request.maximum_tokens,
            dispatch.prefill().output_tiles_per_workgroup(),
            maximum_tiles=16,
          )
        } else {
          4
        },
      ),
      row_width: if selected {
        16
      } else {
        projection_bounded_row_extent(
          maximum_rows=request.maximum_tokens,
          matrix_rows=m,
          preferred_rows=32,
        ).unwrap()
      },
      output_tiles: if selected {
        dispatch.prefill().output_tiles_per_workgroup()
      } else {
        Int::max(1, dispatch.prefill().output_tiles_per_workgroup() / 2)
      },
      minimum_rows: 2,
    })
  }
}

///|
/// Select a complete group of ordered reduction tiles. Short reductions and
/// non-power-of-two extents use the same fold, without reading padded weights
/// or changing the order of floating-point accumulation.
fn projection_transfer_width(
  extent : Int,
  tile : Int,
  maximum_tiles : Int,
) -> Int {
  for tiles = maximum_tiles; tiles > 1; tiles = tiles / 2 {
    if extent % (tile * tiles) == 0 {
      return tile * tiles
    }
  }
  tile
}

///|
/// Profitability policy for immutable operand lookahead, not a numerical
/// transformation. Short folds and wide independent products keep the compact
/// window. Dense and materialized folds supply different live-storage budgets;
/// a device backend still owns physical transfer and shared-memory lowering.
fn projection_operand_window_tiles(
  reduction_extent : Int,
  row_bound : Int,
  output_tiles : Int,
  maximum_tiles~ : Int,
) -> Int {
  if reduction_extent >= 2048 && row_bound <= 16 && output_tiles <= 2 {
    maximum_tiles
  } else {
    4
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
