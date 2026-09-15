///|
/// Composition of striped vector ownership with a logical operand row view.
/// A device supplies its vector width; operand identity and decision order
/// remain independent of instruction spelling and model family.
pub struct ProjectionWeightTransferPlan {
  priv map : ProjectionVectorTransferMap
  priv selection : ProjectionRowSelection
  priv vectors_per_row : Int
} derive(Debug, Eq)

///|
pub fn ProjectionWeightTransferPlan::new(
  map : ProjectionVectorTransferMap,
  segments : ArrayView[ProjectionRowSegment],
  row_width~ : Int,
  vector_width~ : Int,
) -> ProjectionWeightTransferPlan? {
  guard row_width > 0 && vector_width > 0 && row_width % vector_width == 0 else {
    return None
  }
  let vectors_per_row = row_width / vector_width
  let (vectors, _, _) = map.geometry()
  guard vectors % vectors_per_row == 0 &&
    plan_projection_row_selection(segments) is Some(selection) else {
    return None
  }
  Some({ map, selection, vectors_per_row, })
}

///|
pub fn ProjectionWeightTransferPlan::vector_map(
  self : ProjectionWeightTransferPlan,
) -> ProjectionVectorTransferMap {
  self.map
}

///|
pub fn ProjectionWeightTransferPlan::row_selection(
  self : ProjectionWeightTransferPlan,
) -> ProjectionRowSelection {
  self.selection
}

///|
pub fn ProjectionWeightTransferPlan::row_vectors(
  self : ProjectionWeightTransferPlan,
) -> Int {
  self.vectors_per_row
}
