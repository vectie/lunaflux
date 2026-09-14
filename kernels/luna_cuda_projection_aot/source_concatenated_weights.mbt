///|
/// Device spelling of a generic concatenated row view. Keep the existing
/// nested decision schedule; the pure compiler owns interval geometry.
fn projection_concatenated_weight_selection(
  segments : ArrayView[@luna_projection_tile_compiler.ProjectionRowSegment],
  operands : ArrayView[String],
  logical_row : String,
  pointer : String,
  local_row : String,
) -> (String, String) {
  fn tail(index : Int) -> String {
    if index == segments.length() - 1 {
      "\{pointer} = \{operands[index]};"
    } else {
      let width = segments[index].row_end - segments[index].row_begin
      "if (\{local_row} >= \{width}) { \{local_row} -= \{width}; " +
      tail(index + 1) +
      " } else \{pointer} = \{operands[index]};"
    }
  }
  let first = "const __nv_bfloat16 *\{pointer} = \{operands[0]}; int \{local_row} = \{logical_row};"
  let branch = if segments.length() == 1 {
    ""
  } else {
    let width = segments[0].row_end - segments[0].row_begin
    "if (\{local_row} >= \{width}) { \{local_row} -= \{width}; " +
    tail(1) +
    " }"
  }
  (first, branch)
}
