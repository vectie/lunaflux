///|
/// An ingress epilogue observes one complete head, so its materialized operand
/// product is one row microtile by one head, not the entire Q/K/V concatenation.
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
        row_width: rows,
        output_tiles: spec.head_dimension / columns,
        minimum_rows: 2,
        stages: 2,
        fragments: Some(1),
      },
      MatrixFold,
      request,
    ),
  )
}
