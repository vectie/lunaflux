///|
/// Storage scope is distinct from semantic sharing. Both choices compute the
/// same selected rows and retain the same reduction; their traffic, barriers
/// and workgroup occupancy differ and must be measured as complete schedules.
pub(all) enum ProjectionInputResidency {
  ReductionTileResidency
  SelectedRowTileResidency
} derive(Debug, Eq)

///|
pub fn ProjectionStrategy::input_residency(
  self : ProjectionStrategy,
) -> ProjectionInputResidency {
  self.input_residency
}

///|
pub fn ProjectionStrategy::with_selected_row_residency(
  self : ProjectionStrategy,
  stable_id~ : Int,
) -> ProjectionStrategy raise ProjectionStrategyError {
  if self.kernel_class != MatrixTile {
    raise InvalidProjectionStrategy(Candidate)
  }
  ProjectionStrategy::new(
    stable_id~,
    phase=self.phase,
    strategy_class=Some(self.strategy_class),
    kernel_class=self.kernel_class,
    row_bucket=self.row_bucket,
    tile=self.tile,
    output_tiles_per_workgroup=self.output_tiles_per_workgroup,
    input_residency=SelectedRowTileResidency,
  )
}
