///|
/// A finite parallel map, striped over a statically known set of owners.
/// This describes independent operand transfers, not a reassociation of the
/// ordered matrix fold. Lowering can partially evaluate its loop stride.
pub struct ProjectionVectorTransferMap {
  priv vectors : Int
  priv owners : Int
} derive(Debug, Eq)

///|
pub fn ProjectionVectorTransferMap::new(
  vectors~ : Int,
  owners~ : Int,
) -> ProjectionVectorTransferMap? {
  guard vectors > 0 && owners > 0 else { return None }
  Some({ vectors, owners })
}

///|
pub fn ProjectionVectorTransferMap::geometry(
  self : ProjectionVectorTransferMap,
) -> (Int, Int, Int) {
  (self.vectors, self.owners, (self.vectors - 1) / self.owners + 1)
}

///|
pub fn ProjectionVectorTransferMap::vector_at(
  self : ProjectionVectorTransferMap,
  owner : Int,
  round : Int,
) -> Int? {
  guard owner >= 0 &&
    owner < self.owners &&
    owner < self.vectors &&
    round >= 0 &&
    round <= (self.vectors - 1 - owner) / self.owners else {
    return None
  }
  Some(owner + round * self.owners)
}
