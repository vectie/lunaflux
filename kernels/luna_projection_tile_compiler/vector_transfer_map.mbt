///|
/// A finite parallel map, striped over a statically known set of owners.
/// This describes independent operand transfers, not a reassociation of the
/// ordered matrix fold. Lowering can partially evaluate its loop stride.
pub struct ProjectionVectorTransferMap {
  priv ownership : @physical_tile_ir.VectorOwnership
} derive(Debug, Eq)

///|
pub fn ProjectionVectorTransferMap::new(
  vectors~ : Int,
  owners~ : Int,
) -> ProjectionVectorTransferMap? {
  @physical_tile_ir.VectorOwnership::new(vectors~, owners~).map(ownership => {
    ownership,
  })
}

///|
pub fn ProjectionVectorTransferMap::geometry(
  self : ProjectionVectorTransferMap,
) -> (Int, Int, Int) {
  self.ownership.geometry()
}

///|
pub fn ProjectionVectorTransferMap::vector_at(
  self : ProjectionVectorTransferMap,
  owner : Int,
  round : Int,
) -> Int? {
  self.ownership.vector_at(owner, round)
}
