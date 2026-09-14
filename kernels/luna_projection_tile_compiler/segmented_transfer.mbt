///|
/// One intersection of an immutable operand and a worker-round of a vector
/// map. A worker owns the same vector before and after segmentation. Operand
/// selection and the round offset are static, not predicates in the fold.
pub struct ProjectionTransferSegment {
  operand : Int
  round : Int
  worker_begin : Int
  worker_end : Int
  operand_vector_origin : Int
} derive(Debug, Eq)

///|
/// Pure interval intersection of concatenated operands with round-robin
/// worker ownership. Counts are in vectors, independent of scalar dtype,
/// backend, subgroup width, shared layout, or instruction choice.
/// `max_segments` is a caller-controlled source expansion budget, not a
/// hardware or operand-count limit. None retains the caller's loop realization.
pub fn partition_projection_transfer_segments(
  operand_vectors : ArrayView[Int],
  workers~ : Int,
  max_segments? : Int = 4096,
) -> Array[ProjectionTransferSegment]? {
  if workers <= 0 ||
    max_segments <= 0 ||
    operand_vectors.length() > max_segments {
    return None
  }
  guard plan_projection_concatenated_rows(operand_vectors) is Some(operands) else {
    return None
  }
  // Count exact operand/round intersections before emitting any segment.
  // Aligned boundaries do not incur the extra block assumed by the old
  // conservative estimate. Subtract from the budget to avoid overflow.
  for operand in operands; remaining = max_segments {
    let count = (operand.row_end - 1) / workers -
      operand.row_begin / workers +
      1
    if count > remaining {
      return None
    }
    continue remaining - count
  }
  let result : Array[ProjectionTransferSegment] = []
  for segment in operands {
    let operand = segment.operand
    let origin = segment.row_begin
    let end = segment.row_end
    for round in (origin / workers)..<((end - 1) / workers + 1) {
      let round_origin = round * workers
      result.push({
        operand,
        round,
        worker_begin: Int::max(0, origin - round_origin),
        worker_end: Int::min(workers, end - round_origin),
        operand_vector_origin: round_origin - origin,
      })
    }
  }
  Some(result)
}
