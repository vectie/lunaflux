///|
/// Device-parameterized packing of two immutable matrix operands. Ownership
/// and storage extents are planned together; lowering supplies only its vector
/// width and owner count. No reduction order or synchronization is introduced.
pub struct ProjectionOperandTransferPlan {
  priv input : ProjectionVectorTransferMap
  priv weight : ProjectionVectorTransferMap
  priv input_elements : Int
  priv slot_elements : Int
} derive(Debug, Eq)

///|
/// Construct a complete strip-transfer domain, rejecting truncated vectors and
/// unrepresentable storage before a backend can emit wrapped loop bounds.
pub fn ProjectionOperandTransferPlan::new(
  input_rows~ : Int,
  weight_rows~ : Int,
  row_width~ : Int,
  vector_width~ : Int,
  owners~ : Int,
) -> ProjectionOperandTransferPlan? {
  guard input_rows > 0 &&
    weight_rows > 0 &&
    row_width > 0 &&
    vector_width > 0 &&
    owners > 0 &&
    row_width % vector_width == 0 else {
    return None
  }
  let input_elements = input_rows.to_int64() * row_width.to_int64()
  let weight_elements = weight_rows.to_int64() * row_width.to_int64()
  // Two positive Int products fit Int64 even when their sum would not fit Int.
  let slot_elements = input_elements + weight_elements
  guard slot_elements <= 2147483647L else { return None }
  Some({
    input: ProjectionVectorTransferMap::new(
      vectors=(input_elements / vector_width.to_int64()).to_int(),
      owners~,
    ).unwrap(),
    weight: ProjectionVectorTransferMap::new(
      vectors=(weight_elements / vector_width.to_int64()).to_int(),
      owners~,
    ).unwrap(),
    input_elements: input_elements.to_int(),
    slot_elements: slot_elements.to_int(),
  })
}

///|
pub fn ProjectionOperandTransferPlan::input_map(
  self : ProjectionOperandTransferPlan,
) -> ProjectionVectorTransferMap {
  self.input
}

///|
pub fn ProjectionOperandTransferPlan::weight_map(
  self : ProjectionOperandTransferPlan,
) -> ProjectionVectorTransferMap {
  self.weight
}

///|
/// The weight strip begins immediately after the input strip in each slot.
pub fn ProjectionOperandTransferPlan::weight_offset(
  self : ProjectionOperandTransferPlan,
) -> Int {
  self.input_elements
}

///|
pub fn ProjectionOperandTransferPlan::slot_elements(
  self : ProjectionOperandTransferPlan,
) -> Int {
  self.slot_elements
}
