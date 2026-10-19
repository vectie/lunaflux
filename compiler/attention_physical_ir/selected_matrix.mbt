///|
/// Independent sparse rows with padded matrix instruction rows. This schedule
/// never assumes different queries have identical selected keys. Arithmetic
/// reassociation must be authorized separately by the numerical program.
pub struct SelectedMatrixFold {
  priv columns : Int
  priv reduction : Int
}

///|
pub fn SelectedMatrixFold::new(
  qk_dimension~ : Int,
  value_dimension~ : Int,
  columns~ : Int,
  reduction~ : Int,
) -> Self? {
  guard columns > 0 &&
    reduction > 0 &&
    qk_dimension > 0 &&
    qk_dimension % reduction == 0 &&
    value_dimension > 0 &&
    value_dimension % columns == 0 else {
    return None
  }
  Some({ columns, reduction, })
}

///|
pub fn SelectedMatrixFold::geometry(self : Self) -> (Int, Int) {
  (self.columns, self.reduction)
}

///|
pub fn SelectedMatrixFold::score_tiles(self : Self, keys : Int) -> Int {
  if keys <= 0 {
    0
  } else {
    1 + (keys - 1) / self.columns
  }
}
