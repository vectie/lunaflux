///|
/// Scalar-element counts before backend dtype packing. Register-resident
/// accumulators are not included; this describes the local-memory store/load
/// arena and optional selected-input materialization of this schedule family.
pub struct ProjectionLocalStorage {
  priv accumulator_elements : Int64
  priv selected_input_elements : Int64
} derive(Debug, Eq)

///|
pub fn ProjectionLocalStorage::accumulator_elements(
  self : ProjectionLocalStorage,
) -> Int64 {
  self.accumulator_elements
}

///|
pub fn ProjectionLocalStorage::selected_input_elements(
  self : ProjectionLocalStorage,
) -> Int64 {
  self.selected_input_elements
}

///|
fn checked_storage_product(
  left : Int64,
  right : Int64,
) -> Int64 raise ProjectionStrategyError {
  if left <= 0L || right <= 0L || left > 9223372036854775807L / right {
    raise InvalidProjectionStrategy(Capability)
  }
  left * right
}

///|
pub fn ProjectionStrategy::local_storage(
  self : ProjectionStrategy,
  shape : ProjectionShape,
) -> ProjectionLocalStorage raise ProjectionStrategyError {
  if self.kernel_class != MatrixTile {
    return { accumulator_elements: 0L, selected_input_elements: 0L }
  }
  let rows = self.tile.rows.to_int64()
  let accumulator_elements = checked_storage_product(
    checked_storage_product(rows, self.tile.columns.to_int64()),
    self.output_tiles_per_workgroup.to_int64(),
  )
  let selected_input_elements = if shape.family == LanguageModelHead {
    checked_storage_product(
      rows,
      (if self.input_residency == SelectedRowTileResidency {
        shape.input_width
      } else {
        self.tile.reduction
      }).to_int64(),
    )
  } else {
    0L
  }
  { accumulator_elements, selected_input_elements }
}
