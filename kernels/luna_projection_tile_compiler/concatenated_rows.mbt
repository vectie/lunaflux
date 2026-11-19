///|
/// A half-open row interval in a logical concatenation of immutable operands.
/// Operand identity is an ordinal, not a model name or a backend pointer.
pub struct ProjectionRowSegment {
  operand : Int32
  row_begin : Int32
  row_end : Int32
} derive(Debug, Eq)

///|
/// Prefix-sum planning for a concatenated row view. Physical buffers remain
/// separate; a lowering maps an interval to its corresponding ABI operand.
/// Empty operands and unrepresentable logical extents are rejected.
pub fn plan_projection_concatenated_rows(
  row_counts : ArrayView[Int32],
) -> ReadOnlyArray[ProjectionRowSegment]? {
  if row_counts.length() == 0 {
    return None
  }
  let segments : Array[ProjectionRowSegment] = []
  for operand in 0..<row_counts.length(); begin = 0 {
    let count = row_counts[operand]
    if count <= 0 || begin > 2147483647 - count {
      return None
    }
    let end = begin + count
    segments.push({ operand, row_begin: begin, row_end: end, })
    continue end
  }
  Some(ReadOnlyArray::from_array(segments))
}

///|
pub extend ProjectionRowSegment with @debug.Debug::{to_repr}

///|
pub extend ProjectionRowSegment with Eq::{not_equal, equal}
