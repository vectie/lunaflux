///|
/// Cooperative copy followed by independent ordered score folds. The copy
/// layout is bijective; it changes physical ownership, never a reduction tree.
pub struct SelectedScoreTile {
  priv keys : Int
  priv components : Int
}

///|
pub fn SelectedSparseSchedule::score_tile(
  self : Self,
  components : Int,
) -> SelectedScoreTile? {
  guard components > 0 &&
    self.workers > 0 &&
    (self.workers & (self.workers - 1)) == 0 &&
    components <= self.workers / 2 &&
    self.slots >= self.workers else {
    return None
  }
  Some({ keys: self.workers, components, })
}

///|
pub fn SelectedScoreTile::geometry(self : Self) -> (Int, Int) {
  (self.keys, self.components)
}

///|
pub fn SelectedScoreTile::storage_cells(self : Self) -> Int64 {
  self.keys.to_int64() * self.components.to_int64() + self.components.to_int64()
}

///|
pub fn SelectedScoreTile::cell(
  self : Self,
  key : Int,
  component : Int,
) -> Int64? {
  guard key >= 0 &&
    key < self.keys &&
    component >= 0 &&
    component < self.components else {
    return None
  }
  Some(
    component.to_int64() * self.keys.to_int64() +
    (key ^ (component * 2)).to_int64(),
  )
}
