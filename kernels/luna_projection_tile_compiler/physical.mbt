///|
/// Refine an ordered tile fold into explicit shared-fragment reads and SSA
/// product dependencies. CUDA syntax is not an input to this boundary.
pub fn ProjectionFoldPipeline::fragment_program(
  self : ProjectionFoldPipeline,
  rows~ : Int,
  columns~ : Int,
) -> @physical_tile_ir.FragmentProgram? {
  @physical_tile_ir.FragmentProgram::new(
    rows~,
    columns~,
    extent=self.transfer_width(),
    quantum=self.tile_width(),
    slots=self.fragment_stage_count(),
  )
}
