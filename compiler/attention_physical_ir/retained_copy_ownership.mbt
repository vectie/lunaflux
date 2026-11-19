///|
/// Finite ownership of copy addresses retained between two operand transfers.
/// Each owner has a statically named binding per slot, rather than an indexed
/// mutable array whose materialization is left to a device compiler.
pub struct RetainedCopyOwnership {
  priv vectors : Int32
  priv owners : Int32
  priv slots : Int32
} derive(Debug, Eq)

///|
pub extend RetainedCopyOwnership with Eq::{equal, not_equal}

///|
pub extend RetainedCopyOwnership with @debug.Debug::{to_repr}

///|
pub fn RetainedCopyOwnership::new(
  vectors~ : Int32,
  owners~ : Int32,
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
) -> (Int32, Int32, Int32) {
  (self.vectors, self.owners, self.slots)
}

///|
pub fn RetainedCopyOwnership::vector_index(
  self : RetainedCopyOwnership,
  slot : Int32,
  owner : Int32,
) -> Int32? {
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
