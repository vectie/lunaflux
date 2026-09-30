///|
/// Finite ownership of copy addresses retained between two operand transfers.
/// Each owner has a statically named binding per slot, rather than an indexed
/// mutable array whose materialization is left to a device compiler.
pub struct RetainedCopyOwnership {
  priv vectors : Int
  priv owners : Int
  priv slots : Int
} derive(Debug, Eq)

///|
pub extend RetainedCopyOwnership with Eq::{equal, not_equal}

///|
pub extend RetainedCopyOwnership with @debug.Debug::{to_repr}

///|
pub fn RetainedCopyOwnership::new(
  vectors~ : Int,
  owners~ : Int,
) -> RetainedCopyOwnership? {
  if vectors <= 0 || owners <= 0 {
    None
  } else {
    Some({
      vectors,
      owners,
      slots: vectors / owners + (if vectors % owners == 0 { 0 } else { 1 }),
    })
  }
}

///|
pub fn RetainedCopyOwnership::dimensions(
  self : RetainedCopyOwnership,
) -> (Int, Int, Int) {
  (self.vectors, self.owners, self.slots)
}

///|
pub fn RetainedCopyOwnership::vector_index(
  self : RetainedCopyOwnership,
  slot : Int,
  owner : Int,
) -> Int? {
  if slot < 0 || slot >= self.slots || owner < 0 || owner >= self.owners {
    None
  } else {
    let vector = slot.to_int64() * self.owners.to_int64() + owner.to_int64()
    if vector >= self.vectors.to_int64() {
      None
    } else {
      Some(vector.to_int())
    }
  }
}
