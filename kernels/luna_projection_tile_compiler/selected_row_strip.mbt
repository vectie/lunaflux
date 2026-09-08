///|
// Strip mining changes operand lifetimes, not the ordered reduction tree.
// The plan is device-neutral; vector loads and matrix instructions belong to
// the lowering. A single output tile gives each workgroup exclusive storage.
pub struct ProjectionSelectedRowStrip {
  rows : Int
  columns : Int
  reduction : Int
}

///|
pub fn projection_selected_row_strip(
  dispatch : @luna_projection_strategy.ProjectionDispatchPlan,
  demand : ProjectionResultDemand,
) -> ProjectionSelectedRowStrip? {
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
    input_width < 128 ||
    input_width % 128 != 0 ||
    output_width % 16 != 0 {
    None
  } else {
    Some({ rows, columns, reduction: 128 })
  }
}

///|
pub fn ProjectionSelectedRowStrip::reduction_elements(
  self : ProjectionSelectedRowStrip,
) -> Int {
  self.reduction
}

///|
pub fn ProjectionSelectedRowStrip::local_storage_bytes(
  self : ProjectionSelectedRowStrip,
) -> Int {
  (self.rows + self.columns) * self.reduction * 2 + self.rows * self.columns * 4
}

///|
pub fn ScheduledProjectionTileCompilation::selected_row_strip(
  self : ScheduledProjectionTileCompilation,
) -> ProjectionSelectedRowStrip? {
  let request = self.optimized.semantic.selected.request
  projection_selected_row_strip(request.dispatch, request.demand)
}
