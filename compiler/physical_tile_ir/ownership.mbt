///|
/// Device-neutral striped ownership of complete vectors. The extent is finite;
/// the descriptor is constant-sized even for a large transfer domain.
pub struct VectorOwnership {
  priv vectors : Int
  priv owners : Int
} derive(Eq, Debug)

///|
pub extend VectorOwnership with Eq::{equal, not_equal}

///|
pub extend VectorOwnership with @debug.Debug::{to_repr}

///|
pub fn VectorOwnership::new(vectors~ : Int, owners~ : Int) -> VectorOwnership? {
  guard vectors > 0 && owners > 0 else { return None }
  Some({ vectors, owners, })
}

///|
pub fn VectorOwnership::geometry(self : VectorOwnership) -> (Int, Int, Int) {
  (self.vectors, self.owners, (self.vectors - 1) / self.owners + 1)
}

///|
pub fn VectorOwnership::vector_at(
  self : VectorOwnership,
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
