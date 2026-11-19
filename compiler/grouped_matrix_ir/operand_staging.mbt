///|
/// Physical operand lifetimes, independent of dot-product arithmetic. Buffered
/// staging overlaps publication of the next tile with reads of the current
/// tile. Each consumer still visits the same ascending reduction fragments.
pub(all) enum MatrixStaging {
  Synchronous
  SynchronousTiles(inner_tile~ : Int32)
  WorksetBoundedTiles(
    inner_tile~ : Int32,
    maximum_weight_bytes~ : Int64,
    maximum_tiles~ : Int32
  )
  Buffered(inner_tile~ : Int32)
  FragmentPipeline(row_tile~ : Int32)
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
pub fn MatrixTilePlan::operand_slots(self : Self) -> Int32 {
  match self.staging {
    Synchronous | SynchronousTiles(_) | WorksetBoundedTiles(..) => 1
    Buffered(_) | FragmentPipeline(_) => 2
  }
}
