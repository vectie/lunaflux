///|
fn CudaProjectionTransportLayout::plan(
  self : CudaProjectionTransportLayout,
) -> @physical_tile_ir.OperandLayout {
  self.checked_plan().unwrap()
}

///|
/// Fixtures enter through the mandatory physical pass; production emission
/// cannot accept a merely scheduled compilation.
fn projection_source(
  symbol : String,
  parameters : ProjectionParameters,
  maximum_tokens : Int,
  maximum_rows : Int,
  dispatch : @luna_projection_strategy.ProjectionDispatchPlan,
  compilation : @luna_projection_tile_compiler.ScheduledProjectionTileCompilation,
  embed : Bool,
  shared : Bool,
  max_registers~ : Int,
) -> Bytes raise ProjectionAotError {
  projection_emit_source(
    symbol,
    parameters,
    maximum_tokens,
    maximum_rows,
    dispatch,
    projection_refine_physical(compilation),
    embed,
    shared,
    max_registers~,
  )
}

///|
fn physical_pipeline_fixture(
  pipeline : @luna_projection_tile_compiler.ProjectionFoldPipeline,
  groups : Int,
  products? : Int = 1,
  execution? : @luna_projection_tile_compiler.ProjectionTransferExecution,
) -> @luna_projection_tile_compiler.ProjectionPhysicalPipeline {
  @luna_projection_tile_compiler.ProjectionPhysicalPipeline::new(
    pipeline,
    ({ columns: pipeline.transfer_width(), } : CudaProjectionTransportLayout).plan(),
    consumer_groups=groups,
    owners_per_group=32,
    products~,
    execution.unwrap_or(
      if projection_operand_copy_is_async(pipeline.transfer_width()) {
        OverlappedTransfer
      } else {
        CompletedPrefetch
      },
    ),
  ).unwrap()
}

///|
fn dot_fold_fixture(
  extent : Int,
  dots : ArrayView[CudaProjectionDot],
  reduction : @luna_projection_tile_compiler.ProjectionSingleRowReduction,
) -> String {
  projection_dot_fold_source(
    @luna_projection_tile_compiler.ProjectionScalarFoldPlan::new(
      extent~,
      products=dots.length(),
      reduction,
    ).unwrap(),
    dots,
  )
}

///|
fn fold_pipeline_fixture(
  pipeline : @luna_projection_tile_compiler.ProjectionFoldPipeline,
  input : String,
  weight : String,
  stride : Int,
  groups : Int,
  output : Int,
) -> String {
  let physical = @luna_projection_tile_compiler.ProjectionPhysicalPipeline::new(
    pipeline,
    ({ columns: pipeline.transfer_width(), } : CudaProjectionTransportLayout).plan(),
    consumer_groups=groups,
    owners_per_group=32,
    products=1,
    output_tiles=groups,
    if projection_operand_copy_is_async(pipeline.transfer_width()) {
      OverlappedTransfer
    } else {
      CompletedPrefetch
    },
  ).unwrap()
  projection_fold_pipeline_source(physical, input, weight, stride, output)
}

///|
fn projection_concatenated_weight_selection(
  segments : ArrayView[@luna_projection_tile_compiler.ProjectionRowSegment],
  operands : ArrayView[String],
  logical_row : String,
  pointer : String,
  local_row : String,
) -> (String, String) {
  projection_weight_selection_source(
    @luna_projection_tile_compiler.plan_projection_row_selection(segments).unwrap(),
    operands,
    logical_row,
    pointer,
    local_row,
  )
}

///|
fn phased_transfer_fixture(
  operands : @physical_tile_ir.OperandStorage,
  input_width : Int,
) -> String {
  let segments = @luna_projection_tile_compiler.partition_projection_transfer_segments(
    [
      operands.region(0).unwrap().ownership().geometry().0,
      operands.region(1).unwrap().ownership().geometry().0,
      operands.region(2).unwrap().ownership().geometry().0,
    ],
    workers=operands.region(0).unwrap().ownership().geometry().1,
  ).unwrap()
  projection_sibling_phased_transfer_source(operands, segments, input_width)
}
