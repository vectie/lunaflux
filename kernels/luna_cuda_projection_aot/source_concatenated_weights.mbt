///|
/// Device spelling of a generic concatenated row view. Keep the existing
/// nested decision schedule; the pure compiler owns its decision tree.
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
fn projection_weight_selection_source(
  tree : @luna_projection_tile_compiler.ProjectionRowSelection,
  operands : ArrayView[String],
  logical_row : String,
  pointer : String,
  local_row : String,
) -> (String, String) {
  fn render(
    tree : @luna_projection_tile_compiler.ProjectionRowSelection,
  ) -> String {
    match tree {
      Operand(index) => "\{pointer} = \{operands[index]};"
      Skip(width~, operand~, next) =>
        "if (\{local_row} >= \{width}) { \{local_row} -= \{width}; " +
        render(next) +
        " } else \{pointer} = \{operands[operand]};"
    }
  }
  let first = "const __nv_bfloat16 *\{pointer} = \{operands[0]}; int \{local_row} = \{logical_row};"
  let branch = match tree {
    Operand(_) => ""
    Skip(width~, operand=_, next) =>
      "if (\{local_row} >= \{width}) { \{local_row} -= \{width}; " +
      render(next) +
      " }"
  }
  (first, branch)
}
