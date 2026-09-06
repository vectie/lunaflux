///|
/// Enumerates output-map partitions of an already legal matrix strategy.
/// Every candidate retains the same row/reduction tiles and numeric order.
/// Offline measurement keys include the complete distribution, shape, target
/// and backend. No benchmark or search runs during request dispatch.
pub fn ProjectionStrategy::output_distribution_candidates(
  self : ProjectionStrategy,
) -> ReadOnlyArray[ProjectionStrategy] raise ProjectionStrategyError {
  if self.kernel_class != MatrixTile {
    return [self]
  }
  let candidates : Array[ProjectionStrategy] = [self]
  for group in [1, 2, 4, 8, 16] {
    if group != self.output_tiles_per_workgroup {
      candidates.push(
        ProjectionStrategy::new(
          stable_id=if self.stable_id == 1000 + group {
            2000 + group
          } else {
            1000 + group
          },
          phase=self.phase,
          strategy_class=Some(self.strategy_class),
          kernel_class=self.kernel_class,
          row_bucket=self.row_bucket,
          tile=self.tile,
          output_tiles_per_workgroup=group,
          input_residency=self.input_residency,
        ),
      )
    }
  }
  ReadOnlyArray::from_array(candidates)
}
