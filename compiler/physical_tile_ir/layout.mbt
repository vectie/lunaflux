///|
/// Compact row-preserving XOR layout. All parameters are device-independent;
/// selecting them for bank geometry is a backend responsibility. A producer
/// and consumer must bind the same value, not separate address formulas.
pub struct OperandLayout {
  priv columns : Int
  priv vector : Int
  priv rotations : Int
  priv row_group : Int
} derive(Eq, Debug)

///|
pub extend OperandLayout with Eq::{equal, not_equal}

///|
pub extend OperandLayout with @debug.Debug::{to_repr}

///|
pub fn OperandLayout::new(
  columns~ : Int,
  vector~ : Int,
  rotations~ : Int,
  row_group~ : Int,
) -> OperandLayout? {
  guard columns > 0 &&
    vector > 0 &&
    rotations > 0 &&
    row_group > 0 &&
    (vector & (vector - 1)) == 0 &&
    (rotations & (rotations - 1)) == 0 &&
    vector.to_int64() * rotations.to_int64() <= columns.to_int64() &&
    columns.to_int64() % (vector.to_int64() * rotations.to_int64()) == 0L else {
    return None
  }
  Some({ columns, vector, rotations, row_group, })
}

///|
pub fn OperandLayout::geometry(self : OperandLayout) -> (Int, Int, Int, Int) {
  (self.columns, self.vector, self.rotations, self.row_group)
}

///|
pub fn OperandLayout::offset(
  self : OperandLayout,
  row : Int,
  column : Int,
) -> Int? {
  guard row >= 0 && column >= 0 && column < self.columns else { return None }
  let mask = row / self.row_group % self.rotations * self.vector
  let offset = row.to_int64() * self.columns.to_int64() +
    (column ^ mask).to_int64()
  if offset > 2147483647L {
    None
  } else {
    Some(offset.to_int())
  }
}

///|
pub fn OperandLayout::canonical(self : OperandLayout) -> String {
  "tile-operand-layout-v1;columns=\{self.columns};vector=\{self.vector};rotations=\{self.rotations};row-group=\{self.row_group}"
}
