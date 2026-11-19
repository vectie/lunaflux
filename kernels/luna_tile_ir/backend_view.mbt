///|
/// Read-only backend view of the already validated canonical program.
pub fn LunaTileProgram::tensors(
  self : Self,
) -> ReadOnlyArray[LunaTileTensorView] {
  self.tensors
}

///|
pub fn LunaTileProgram::instructions(
  self : Self,
) -> ReadOnlyArray[LunaTileInstruction] {
  self.instructions
}

///|
pub fn LunaTileProgram::loop_extents(self : Self) -> ReadOnlyArray[Int64] {
  self.loop_extents
}

///|
pub fn LunaTileRegion::origins(
  self : Self,
) -> ReadOnlyArray[LunaTileAffineExpr] {
  self.origins
}

///|
pub fn LunaTileRegion::extents(self : Self) -> ReadOnlyArray[Int64] {
  self.extents
}
