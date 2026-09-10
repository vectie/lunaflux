///|
/// A selected-row matrix value with one consumer group can be gathered into
/// that group's operand fragments. No local-memory materialization is shared
/// with another consumer. This changes value placement, not the ordered fold.
pub struct ProjectionSelectedRowGather {
  rows : Int
  columns : Int
  reduction : Int
}

///|
pub fn projection_selected_row_gather(
  dispatch : @luna_projection_strategy.ProjectionDispatchPlan,
  demand : ProjectionResultDemand,
) -> ProjectionSelectedRowGather? {
  let strategy = dispatch.prefill()
  let (rows, columns, reduction) = strategy.tile().dimensions()
  let (input_width, output_width, _) = dispatch.shape().dimensions()
  if demand != QueryRowEnds ||
    dispatch.shape().family() != LanguageModelHead ||
    strategy.kernel_class() != MatrixTile ||
    strategy.output_tiles_per_workgroup() != 1 ||
    strategy.input_residency() == SelectedRowTileResidency ||
    rows != 16 ||
    columns != 16 ||
    reduction != 16 ||
    input_width % reduction != 0 ||
    output_width % columns != 0 {
    None
  } else {
    Some({ rows, columns, reduction })
  }
}

///|
pub fn ProjectionSelectedRowGather::dimensions(
  self : ProjectionSelectedRowGather,
) -> (Int, Int, Int) {
  (self.rows, self.columns, self.reduction)
}

///|
pub fn ScheduledProjectionTileCompilation::selected_row_gather(
  self : ScheduledProjectionTileCompilation,
) -> ProjectionSelectedRowGather? {
  let request = self.optimized.semantic.selected.request
  projection_selected_row_gather(request.dispatch, request.demand)
}
