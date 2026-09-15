///|
/// Ordered selection in a concatenated operand view. A skip subtracts the
/// current operand extent before evaluating its continuation. Device lowering
/// chooses syntax, not the decision order or row-coordinate transformation.
pub enum ProjectionRowSelection {
  Operand(Int)
  Skip(width~ : Int, operand~ : Int, ProjectionRowSelection)
} derive(Debug, Eq)

///|
/// Build the established linear selection schedule from semantic intervals.
/// This is an AOT transformation, not a token-step lookup or a branch hoist.
pub fn plan_projection_row_selection(
  segments : ArrayView[ProjectionRowSegment],
) -> ProjectionRowSelection? {
  if segments.is_empty() {
    return None
  }
  for i, segment in segments {
    if segment.operand != i ||
      segment.row_end <= segment.row_begin ||
      segment.row_begin != (if i == 0 { 0 } else { segments[i - 1].row_end }) {
      return None
    }
  }
  // The accumulated tree covers the suffix beginning at i + 1; each step
  // prepends exactly one interval, preserving its relative row coordinates.
  for i = segments.length() - 2, tree = Operand(segments.length() - 1); i >= 0; {
    let segment = segments[i]
    continue i - 1,
      Skip(
        width=segment.row_end - segment.row_begin,
        operand=segment.operand,
        tree,
      )
  } nobreak {
    Some(tree)
  }
}
