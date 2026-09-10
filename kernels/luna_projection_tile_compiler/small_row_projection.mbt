///|
/// Bound an immutable row map before storage and product-fold distribution.
/// Only complete matrix microtiles are retained; this does not change the
/// reduction topology, arithmetic order, or actual-row masking. The bound is
/// semantic (tokens for a dense map, selected rows for a gather), not a model
/// family, device generation, or launch dimension.
pub fn projection_bounded_row_extent(
  maximum_rows~ : Int,
  matrix_rows~ : Int,
  preferred_rows~ : Int,
) -> Int? {
  if maximum_rows <= 0 ||
    matrix_rows <= 0 ||
    preferred_rows <= 0 ||
    preferred_rows % matrix_rows != 0 {
    None
  } else if maximum_rows >= preferred_rows {
    Some(preferred_rows)
  } else {
    // maximum_rows < preferred_rows, so this cannot exceed the already
    // representable aligned preferred extent (unlike n + tile - 1).
    Some(((maximum_rows - 1) / matrix_rows + 1) * matrix_rows)
  }
}
