///|
/// Cardinality of the independent row/column product after tiling. Reduction
/// depth and consumer count do not participate: they describe work inside a
/// workgroup, not the number of independent workgroups. This pure projection
/// is shared by offline launch planning and the emitted product-fold loops.
pub fn ProjectionFoldPipeline::workgroup_count(
  self : ProjectionFoldPipeline,
  maximum_rows~ : Int,
  output_columns~ : Int,
  matrix_columns~ : Int,
) -> Int? {
  if maximum_rows <= 0 ||
    output_columns <= 0 ||
    matrix_columns <= 0 ||
    self.row_width <= 0 ||
    self.output_tiles <= 0 {
    return None
  }
  let columns = self.output_tiles.to_int64() * matrix_columns.to_int64()
  let row_groups = (maximum_rows.to_int64() - 1L) / self.row_width.to_int64() +
    1L
  let column_groups = (output_columns.to_int64() - 1L) / columns + 1L
  let count = row_groups * column_groups
  if count > 2147483647L {
    None
  } else {
    Some(count.to_int())
  }
}
