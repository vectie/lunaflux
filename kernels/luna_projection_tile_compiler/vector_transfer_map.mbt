///|
/// A finite parallel map, striped over a statically known set of owners.
/// This describes independent operand transfers, not a reassociation of the
/// ordered matrix fold. Lowering can partially evaluate its loop stride.
pub struct ProjectionVectorTransferMap {
  priv ownership : @physical_tile_ir.VectorOwnership
} derive(Debug, Eq)

///|
pub fn ProjectionVectorTransferMap::new(
  vectors~ : Int32,
  owners~ : Int32,
) -> ProjectionVectorTransferMap? {
  @physical_tile_ir.VectorOwnership::new(vectors~, owners~).map(ownership => {
    ownership,
  })
}

///|
pub fn ProjectionVectorTransferMap::geometry(
  self : ProjectionVectorTransferMap,
) -> (Int32, Int32, Int32) {
  self.ownership.geometry()
}

///|
pub fn ProjectionVectorTransferMap::vector_at(
  self : ProjectionVectorTransferMap,
  owner : Int32,
  round : Int32,
) -> Int32? {
  self.ownership.vector_at(owner, round)
}

///|
pub extend ProjectionVectorTransferMap with @debug.Debug::{to_repr}

///|
pub extend ProjectionVectorTransferMap with Eq::{not_equal, equal}
