///|
/// An ingress epilogue observes one complete head, so its materialized operand
/// product is an independently selected row tile by a group of complete heads.
/// Reuse the same finite ring and ordered fragment fold as ordinary projection.
fn projection_ingress_pipeline(
  optimized : OptimizedProjectionTileCompilation,
) -> ProjectionFoldPipeline? {
  let request = optimized.semantic.selected.request
  guard request.epilogue is ProjectionAttentionIngress(spec) else {
    return None
  }
  let dispatch = request.dispatch
  let (extent, _, _) = dispatch.shape().dimensions()
  let (rows, columns, reduction) = dispatch.prefill().tile().dimensions()
  guard dispatch.prefill().kernel_class() == MatrixTile &&
    rows > 0 &&
    columns > 0 &&
    reduction > 0 &&
    extent > 0 &&
    extent % reduction == 0 &&
    spec.head_dimension % columns == 0 else {
    return None
  }
  Some(
    projection_apply_fold_choice(
      {
        tile_width: reduction,
        tile_count: extent / reduction,
        transfer_width: projection_transfer_width(extent, reduction, 4),
        row_width: rows * request.ingress_row_tiles,
        output_tiles: spec.head_dimension / columns * request.ingress_head_tiles,
        minimum_rows: 2,
        stages: 2,
        fragments: Some(1),
        hoist_addresses: false,
      },
      MatrixFold,
      request,
    ),
  )
}

///|
/// Pack complete, independent head folds into one producer domain. This only
/// changes ownership: each head retains its original dot and epilogue order.
pub fn ProjectionTileCompilerRequest::with_ingress_head_tiles(
  self : ProjectionTileCompilerRequest,
  heads : Int,
) -> ProjectionTileCompilerRequest raise ProjectionTileCompilerError {
  guard self.epilogue is ProjectionAttentionIngress(spec) &&
    heads > 0 &&
    heads <= 2147483647 / spec.head_dimension else {
    raise InvalidProjectionTileCompilation(InvalidProblem)
  }
  { ..self, ingress_head_tiles: heads, }
}

///|
/// Widen the independent row map without changing instruction geometry or
/// the ordered dot fold. Epilogue ownership remains a complete head per row.
pub fn ProjectionTileCompilerRequest::with_ingress_row_tiles(
  self : ProjectionTileCompilerRequest,
  tiles : Int,
) -> ProjectionTileCompilerRequest raise ProjectionTileCompilerError {
  let rows = self.dispatch.prefill().tile().dimensions().0
  guard self.epilogue is ProjectionAttentionIngress(_) &&
    tiles > 0 &&
    rows > 0 &&
    tiles <= 2147483647 / rows else {
    raise InvalidProjectionTileCompilation(InvalidProblem)
  }
  { ..self, ingress_row_tiles: tiles, }
}
