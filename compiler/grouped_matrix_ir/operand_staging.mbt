///|
/// Physical operand lifetimes, independent of dot-product arithmetic. Buffered
/// staging overlaps publication of the next tile with reads of the current
/// tile. Each consumer still visits the same ascending reduction fragments.
pub(all) enum MatrixStaging {
  Synchronous
  Buffered(inner_tile~ : Int)
} derive(Eq, Debug)

///|
pub extend MatrixStaging with Eq::{equal, not_equal}

///|
pub extend MatrixStaging with @debug.Debug::{to_repr}

///|
pub fn MatrixTilePlan::staging(self : Self) -> MatrixStaging {
  self.staging
}

///|
pub fn MatrixTilePlan::operand_slots(self : Self) -> Int {
  match self.staging {
    Synchronous => 1
    Buffered(_) => 2
  }
}
