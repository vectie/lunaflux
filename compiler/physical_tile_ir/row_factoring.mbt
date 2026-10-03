///|
/// Pure address factoring for a row-preserving operand layout. Consumers may
/// retain the row origin and permutation mask across ordered column products.
/// This does not change the producer layout, storage lifetime or byte traffic.
pub struct OperandRowFactoring {
  priv layout : OperandLayout
} derive(Eq, Debug)

///|
pub extend OperandRowFactoring with Eq::{equal, not_equal}

///|
pub extend OperandRowFactoring with @debug.Debug::{to_repr}

///|
pub fn OperandLayout::factor_rows(self : OperandLayout) -> OperandRowFactoring {
  { layout: self, }
}

///|
pub fn OperandRowFactoring::layout(self : OperandRowFactoring) -> OperandLayout {
  self.layout
}

///|
/// (linear row origin, row-local XOR permutation mask). The resulting origin
/// is checked once and can serve every column in the retained row domain.
pub fn OperandRowFactoring::row(
  self : OperandRowFactoring,
  row : Int,
) -> (Int, Int)? {
  let (columns, vector, rotations, row_group) = self.layout.geometry()
  guard row >= 0 else { return None }
  let origin = row.to_int64() * columns.to_int64()
  guard origin + columns.to_int64() - 1L <= 2147483647L else { return None }
  Some((origin.to_int(), row / row_group % rotations * vector))
}

///|
pub fn OperandRowFactoring::column(
  self : OperandRowFactoring,
  permutation : Int,
  column : Int,
) -> Int? {
  let (columns, vector, rotations, _) = self.layout.geometry()
  guard column >= 0 &&
    column < columns &&
    permutation >= 0 &&
    permutation < vector * rotations &&
    permutation % vector == 0 else {
    return None
  }
  Some(column ^ permutation)
}
