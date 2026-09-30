///|
/// Backend binding retained before source emission. Every matrix producer,
/// consumer, decode fold and numerical epilogue consumes this single program.
priv struct QwenIngressPhysicalProgram {
  columns : @luna_projection_tile_compiler.ProjectionColumnFoldPlan
  pipeline : @luna_projection_tile_compiler.ProjectionPhysicalPipeline
  numerics : @luna_projection_tile_compiler.AttentionIngressNumerics
  scalar : @luna_projection_tile_compiler.ProjectionScalarFoldPlan
  packing : @luna_projection_tile_compiler.AttentionIngressPacking
  scratch : @physical_tile_ir.SequentialScratch
}

///|
fn qwen_refine_physical(
  compilation : @luna_projection_tile_compiler.ScheduledProjectionTileCompilation,
  policy : @resource_policy.Policy,
) -> QwenIngressPhysicalProgram raise FusedParallelAotError {
  let columns = qwen_ingress_column_plan(compilation, policy~)
  let (groups, _, live, _) = columns.geometry()
  guard compilation.schedule().matrix_pipeline() is Some(schedule) else {
    raise FusedParallelMismatch(CompilerPolicy)
  }
  guard @physical_tile_ir.OperandLayout::new(
      columns=schedule.transfer_width(),
      vector=8,
      rotations=1,
      row_group=1,
    )
    is Some(layout) else {
    raise FusedParallelMismatch(CompilerPolicy)
  }
  guard @luna_projection_tile_compiler.ProjectionPhysicalPipeline::new(
      schedule,
      layout,
      consumer_groups=groups,
      owners_per_group=32,
      products=1,
      OverlappedTransfer,
      columns=live,
    )
    is Some(pipeline) else {
    raise FusedParallelMismatch(CompilerPolicy)
  }
  guard compilation.program().epilogue() is ProjectionAttentionIngress(epilogue) else {
    raise FusedParallelMismatch(CompilerPolicy)
  }
  let numerics = epilogue.numerics(reduction_lanes=32) catch {
    _ => raise FusedParallelMismatch(CompilerPolicy)
  }
  let packing = epilogue.packing(
    heads_per_group=policy.head_tiles(),
    reduction_lanes=32,
  ) catch {
    _ => raise FusedParallelMismatch(CompilerPolicy)
  }
  let extent = schedule.tile_count() * schedule.tile_width()
  guard @luna_projection_tile_compiler.ProjectionScalarFoldPlan::new(
      extent~,
      products=1,
      compilation.schedule().single_row_reduction(),
    )
    is Some(scalar) else {
    raise FusedParallelMismatch(CompilerPolicy)
  }
  let scratch = ingress_scratch_plan(
    schedule.row_width(),
    schedule.output_tiles() * schedule.tile_width(),
    schedule.transfer_width(),
    schedule.stage_count(),
    columns.geometry().3,
  )
  { columns, pipeline, numerics, scalar, packing, scratch, }
}

///|
/// Multiple accumulator windows publish output while subsequent windows still
/// read staged operands. A single window retires all operand readers first.
fn ingress_scratch_plan(
  rows : Int,
  columns : Int,
  transfer : Int,
  stages : Int,
  windows : Int,
) -> @physical_tile_ir.SequentialScratch {
  @physical_tile_ir.SequentialScratch::new(
    producer_bytes=(rows.to_int64() + columns.to_int64()) *
      transfer.to_int64() *
      stages.to_int64() *
      2L,
    consumer_bytes=rows.to_int64() * columns.to_int64() * 4L,
    overlapping=windows != 1,
  ).unwrap()
}
