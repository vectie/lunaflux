///|
/// A conversion boundary is a semantic stage, not a device instruction.
pub(all) enum Conversion {
  DecodeElement
  ApplyLocalScaleF32
  ApplyGlobalScaleF32
  RoundBf16
} derive(Debug, Eq)

///|
/// Reuses the compiler's physical producer/consumer layout. Representation
/// describes global bytes; OperandRegion describes decoded element ownership.
pub struct PhysicalOperand {
  priv plan : OperandPlan
  priv region : @physical.OperandRegion
  priv conversions : ReadOnlyArray[Conversion]
} derive(Debug, Eq)

///|
pub fn OperandPlan::refine_region(
  self : OperandPlan,
  region : @physical.OperandRegion,
) -> PhysicalOperand raise PrecisionPlanError {
  let (_, rows, columns) = region.geometry()
  if self.placement != DecodeInOperandTile ||
    rows.to_int64() * columns.to_int64() * 2L > self.scratch_bytes {
    raise IllegalPlacement
  }
  let rep = self.representation
  let conversions = [Conversion::DecodeElement]
  if rep.scale_element() != Absent {
    conversions.push(ApplyLocalScaleF32)
  }
  if rep.has_global_scale() {
    conversions.push(ApplyGlobalScaleF32)
  }
  conversions.push(RoundBf16)
  { plan: self, region, conversions: ReadOnlyArray::from_array(conversions), }
}

///|
pub fn PhysicalOperand::plan(self : PhysicalOperand) -> OperandPlan {
  self.plan
}

///|
pub fn PhysicalOperand::region(
  self : PhysicalOperand,
) -> @physical.OperandRegion {
  self.region
}

///|
pub fn PhysicalOperand::conversions(
  self : PhysicalOperand,
) -> ReadOnlyArray[Conversion] {
  self.conversions
}

///|
pub fn PhysicalOperand::canonical(self : PhysicalOperand) -> String {
  let (base, rows, columns) = self.region.geometry()
  let (_, owners, _) = self.region.ownership().geometry()
  // Includes the entire address map, not just its size.
  "precision-physical-v1|\{self.plan.canonical()}|\{base}|\{rows}|\{columns}|\{owners}|\{self.region.layout().canonical()}"
}
