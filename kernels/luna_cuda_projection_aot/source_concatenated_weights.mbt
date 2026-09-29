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
