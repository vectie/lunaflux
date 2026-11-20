///|
fn projection_schedule_canonical(
  optimized : OptimizedProjectionTileCompilation,
  loops : ArrayView[ProjectionLoopCombinator],
  row_tail_policy : ProjectionRowTailPolicy,
  single_row_reduction : ProjectionSingleRowReduction,
) -> Bytes {
  let builder = StringBuilder()
  let explicit_tree = optimized.semantic.selected.request.single_row_numerics ==
    AllowDeterministicDotTree
  if explicit_tree {
    builder <+ "luna-projection-tile-schedule-v7\n"
  } else {
    builder <+ "luna-projection-tile-schedule-v6\n"
  }
  builder <+ "program_sha256=\{optimized.normalized.program.digest.0}\n"
  builder <+ "optimization_sha256=\{optimized.optimization.digest.0}\n"
  for schedule_loop in loops {
    let kind = if schedule_loop.kind == ProjectionParallelMap { 1 } else { 2 }
    let axis = match schedule_loop.axis {
      ProjectionRowAxis => 1
      ProjectionOutputColumnAxis => 2
      ProjectionReductionAxis => 3
      ProjectionIntermediateColumnAxis => 4
      ProjectionIntermediateReductionAxis => 5
      ProjectionHeadAxis => 6
      ProjectionHeadComponentAxis => 7
    }
    builder <+
      "loop=\{kind},\{axis},\{schedule_loop.extent},\{schedule_loop.tile}\n"
  }
  builder <+
    "row_tail_policy=\{if row_tail_policy == ZeroPaddedMatrixRowTail { "masked-matrix" } else { "scalar" }}\n"
  let dispatch = optimized.semantic.selected.request.dispatch
  let primary_groups = projection_primary_consumer_groups(
    optimized.semantic.selected.request,
  )
  if primary_groups != dispatch.prefill().output_tiles_per_workgroup() {
    builder <+
      "\{if dispatch.shape().family() == GatedMlp { "sibling" } else { "primary" }}_consumer_groups=\{primary_groups}\n"
  }
  if projection_selected_row_gather(
      dispatch,
      optimized.semantic.selected.request.demand,
    )
    is Some(gather) {
    let (rows, columns, reduction) = gather.dimensions()
    builder <+
      "selected_row_operand_scope=register-fragment,\{rows},\{columns},\{reduction}\n"
    builder <+ "selected_row_ordered_prefetch=\{gather.prefetch_fragments()}\n"
  }
  if dispatch.prefill().input_residency() == SelectedRowTileResidency {
    builder <+ "selected_input_residency=row-tile\n"
  }
  let baseline_groups = if dispatch.shape().family() == GatedMlp &&
    dispatch.prefill().kernel_class() == MatrixTile {
    16
  } else {
    8
  }
  if dispatch.prefill().output_tiles_per_workgroup() != baseline_groups {
    builder <+
      "output_tiles_per_workgroup=\{dispatch.prefill().output_tiles_per_workgroup()}\n"
  }
  if explicit_tree {
    match single_row_reduction {
      OrderedDot => builder <+ "single_row_reduction=ordered\n"
      StridedPairwiseDot(lanes) =>
        builder <+ "single_row_reduction=strided-pairwise,\{lanes}\n"
    }
  }
  builder <+
    "reuse_intermediate_tile=\{optimized.optimization.applied(ReuseIntermediateTileAcrossOutputTiles)}\n"
  if optimized.optimization.applied(ReuseIntermediateTileAcrossOutputTiles) {
    let tiles = projection_intermediate_pipeline(optimized)
      .map(pipeline => pipeline.output_tiles())
      .unwrap_or(
        projection_companion_output_tiles(
          dispatch.prefill().output_tiles_per_workgroup(),
        ),
      )
    builder <+ "companion_output_tiles_per_workgroup=\{tiles}\n"
  }
  let rotary_optimized = optimized.optimization.applied(
    FuseAttentionIngressEpilogue,
  )
  builder <+ "hoist_rotary_inverse_frequencies=\{rotary_optimized}\n"
  builder <+ "fuse_rotary_pairs=\{rotary_optimized}\n"
  if projection_intermediate_pipeline(optimized) is Some(pipeline) {
    builder <+
      "intermediate_operand_pipeline=\{pipeline.stage_count()},\{pipeline.tile_width},\{pipeline.tile_count},\{pipeline.transfer_width},\{pipeline.row_width()},\{pipeline.single_row_outputs()},\{pipeline.minimum_rows()}\n"
    builder <+
      "intermediate_fragment_pipeline=\{pipeline.fragment_stage_count()}\n"
    if pipeline.hoist_addresses {
      builder <+ "intermediate_producer_address_scope=workgroup\n"
    }
  }
  if projection_sibling_pipeline(optimized) is Some(pipeline) {
    builder <+
      "sibling_operand_pipeline=\{pipeline.stage_count()},\{pipeline.tile_width()},\{pipeline.tile_count()},\{pipeline.transfer_width()},\{pipeline.row_width()},\{pipeline.output_tiles()},\{pipeline.minimum_rows()}\n"
    builder <+ "sibling_fragment_pipeline=\{pipeline.fragment_stage_count()}\n"
    let (row_tile, _, _) = dispatch.prefill().tile().dimensions()
    let sibling_reuse = partition_projection_sibling_reuse(
      row_tiles=pipeline.row_width() / row_tile,
      output_tiles=pipeline.output_tiles(),
      consumer_groups=primary_groups,
    )
    if sibling_reuse is Some(reuse) {
      builder <+
        "sibling_axis_reuse=2,\{reuse.row_groups()},\{reuse.output_groups()},\{reuse.rows_per_group()}\n"
    }
    let effective_groups = primary_groups /
      (if sibling_reuse is Some(_) { 2 } else { 1 })
    if partition_projection_product_fold(
        row_tiles=pipeline.row_width() / row_tile,
        output_tiles=pipeline.output_tiles(),
        consumer_groups=effective_groups,
      )
      is Some(distribution) {
      builder <+
        "sibling_product_distribution=\{distribution.row_groups()},\{distribution.output_groups()},\{distribution.rows_per_group()}\n"
    }
  }
  if projection_matrix_pipeline(optimized) is Some(pipeline) {
    builder <+
      "matrix_operand_pipeline=\{pipeline.stage_count()},\{pipeline.tile_width()},\{pipeline.tile_count()},\{pipeline.transfer_width()},\{pipeline.row_width()},\{pipeline.output_tiles()},\{pipeline.minimum_rows()}\n"
    builder <+ "matrix_fragment_pipeline=\{pipeline.fragment_stage_count()}\n"
    if pipeline.hoist_addresses {
      builder <+ "matrix_producer_address_scope=workgroup\n"
    }
  }
  @utf8.encode(builder.to_string())
}

///|
pub fn schedule_projection_tile(
  optimized : OptimizedProjectionTileCompilation,
) -> ScheduledProjectionTileCompilation {
  let request = optimized.semantic.selected.request
  let (input_width, output_width, _) = request.dispatch.shape().dimensions()
  let (tile_rows, tile_columns, tile_reduction) = request.dispatch
    .prefill()
    .tile()
    .dimensions()
  let row_extent = if request.demand == QueryRowEnds {
    request.maximum_rows
  } else {
    request.maximum_tokens
  }
  let row_loop = ProjectionLoopCombinator::{
    kind: ProjectionParallelMap,
    axis: ProjectionRowAxis,
    extent: row_extent,
    tile: tile_rows,
  }
  let loops : Array[ProjectionLoopCombinator] = if request.dispatch
    .shape()
    .family() ==
    GatedMlp {
    let (_, output_width, intermediate_width) = request.dispatch
      .shape()
      .dimensions()
    [
      row_loop,
      {
        kind: ProjectionParallelMap,
        axis: ProjectionIntermediateColumnAxis,
        extent: intermediate_width,
        tile: tile_columns,
      },
      {
        kind: ProjectionSequentialFold,
        axis: ProjectionReductionAxis,
        extent: input_width,
        tile: tile_reduction,
      },
      {
        kind: ProjectionParallelMap,
        axis: ProjectionOutputColumnAxis,
        extent: output_width,
        tile: tile_columns,
      },
      {
        kind: ProjectionSequentialFold,
        axis: ProjectionIntermediateReductionAxis,
        extent: intermediate_width,
        tile: tile_reduction,
      },
    ]
  } else {
    [
      row_loop,
      {
        kind: ProjectionParallelMap,
        axis: ProjectionOutputColumnAxis,
        extent: output_width,
        tile: tile_columns,
      },
      {
        kind: ProjectionSequentialFold,
        axis: ProjectionReductionAxis,
        extent: input_width,
        tile: tile_reduction,
      },
    ]
  }
  match request.epilogue {
    ProjectionAttentionIngress(spec) => {
      loops.push({
        kind: ProjectionParallelMap,
        axis: ProjectionHeadAxis,
        extent: spec.query_heads + 2 * spec.kv_heads,
        tile: 1,
      })
      loops.push({
        kind: ProjectionParallelMap,
        axis: ProjectionHeadComponentAxis,
        extent: spec.head_dimension,
        tile: spec.head_dimension,
      })
    }
    ProjectionStoreRows => ()
  }
  // Operand reuse is described by the fold lifetime plans, not a second,
  // permanently-false backend toggle. All matrix folds mask their row tails.
  let row_tail_policy = if request.dispatch.prefill().kernel_class() ==
    MatrixTile {
    ZeroPaddedMatrixRowTail
  } else {
    ScalarRowTail
  }
  let single_row_reduction = projection_single_row_reduction(request)
  let canonical = projection_schedule_canonical(
    optimized, loops, row_tail_policy, single_row_reduction,
  )
  let schedule = ProjectionTileSchedule::{
    program_digest: optimized.normalized.program.digest,
    optimization_digest: optimized.optimization.digest,
    demand: request.demand,
    loops: ReadOnlyArray::from_array(loops),
    output_tiles_per_workgroup: projection_primary_consumer_groups(request),
    selected_input_row_resident: request.demand == QueryRowEnds &&
    request.dispatch.prefill().input_residency() == SelectedRowTileResidency,
    row_tail_policy,
    single_row_reduction,
    intermediate_pipeline: projection_intermediate_pipeline(optimized),
    sibling_pipeline: projection_sibling_pipeline(optimized),
    matrix_pipeline: projection_matrix_pipeline(optimized),
    reuse_intermediate_tile: optimized.optimization.applied(
      ReuseIntermediateTileAcrossOutputTiles,
    ),
    fuse_attention_ingress_epilogue: optimized.optimization.applied(
      FuseAttentionIngressEpilogue,
    ),
    hoist_rotary_inverse_frequencies: optimized.optimization.applied(
      HoistRotaryInverseFrequencies,
    ),
    fuse_rotary_pairs: optimized.optimization.applied(FuseRotaryPairEvaluation),
    canonical,
    digest: ProjectionTileScheduleDigest(projection_tile_sha256(canonical)),
  }
  let compilation_canonical = @utf8.encode(
    "luna-projection-functional-compilation-v4\n" +
    "program_sha256=" +
    optimized.normalized.program.digest.0 +
    "\noptimization_sha256=" +
    optimized.optimization.digest.0 +
    "\nschedule_sha256=" +
    schedule.digest.0 +
    "\n",
  )
  {
    optimized,
    schedule,
    canonical: compilation_canonical,
    digest: ProjectionTileCompilationDigest(
      projection_tile_sha256(compilation_canonical),
    ),
  }
}
