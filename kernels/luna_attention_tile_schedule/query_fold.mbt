///|
/// Partially evaluate a subgroup-strided query map into a finite product of
/// independent ordered folds. Offset i owns queries subgroup + offset[i].
/// The runtime active-row predicate remains on each product component; the
/// sequential key fold and its floating-point evaluation order do not change.
pub fn scalarize_attention_query_fold(
  query_tile : Int,
  subgroups : Int,
) -> ReadOnlyArray[Int] raise AttentionTileScheduleError {
  if query_tile <= 0 || query_tile > 64 || subgroups <= 0 || subgroups > 32 {
    raise InvalidAttentionTileSchedule(InvalidTile)
  }
  let slots = (query_tile + subgroups - 1) / subgroups
  [
    for slot in 0..<slots => slot * subgroups
  ]
}

///|
/// A finite pure score map has two consumers (maximum and exponential sum).
/// Materialize its per-lane tuple once within the key-tile fold, preserving
/// increasing-key evaluation order and leaving the tail predicate explicit.
pub fn scalarize_attention_score_map(
  key_tile : Int,
  reduction_lanes : Int,
) -> ReadOnlyArray[Int] raise AttentionTileScheduleError {
  if key_tile <= 0 ||
    key_tile > 128 ||
    reduction_lanes <= 0 ||
    reduction_lanes > 32 {
    raise InvalidAttentionTileSchedule(InvalidTile)
  }
  let slots = (key_tile + reduction_lanes - 1) / reduction_lanes
  [
    for slot in 0..<slots => slot * reduction_lanes
  ]
}
