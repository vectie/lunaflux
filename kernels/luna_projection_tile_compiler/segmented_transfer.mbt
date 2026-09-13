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
pub fn partition_projection_transfer_segments(
  operand_vectors : ArrayView[Int],
  workers~ : Int,
) -> Array[ProjectionTransferSegment]? {
  if workers <= 0 ||
    operand_vectors.length() == 0 ||
    operand_vectors.length() > 128 {
    return None
  }
  let total = for total = 0, i = 0; i < operand_vectors.length(); {
    let count = operand_vectors[i]
    if count <= 0 || total > 2147483647 - count {
      return None
    }
    continue total + count, i + 1
  } nobreak {
    total
  }
  let result : Array[ProjectionTransferSegment] = []
  // A finite compile-time unroll budget, not a shape or backend restriction.
  // Very large maps must use the loop realization instead of emitting millions
  // of static source blocks.
  if (total - 1) / workers + operand_vectors.length() > 4096 {
    return None
  }
  for operand = 0, origin = 0; operand < operand_vectors.length(); {
    let end = origin + operand_vectors[operand]
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
    continue operand + 1, end
  }
  // All arithmetic above is bounded by the checked concatenated extent.
  if total <= 0 {
    None
  } else {
    Some(result)
  }
}
