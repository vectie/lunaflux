///|
/// A pure bijection from a rectangular operand to compact square microtiles.
/// It changes storage order only: each inner reduction keeps its logical order.
pub struct AttentionOperandTileLayout {
  rows : Int32
  columns : Int32
  side : Int32
} derive(Eq, Debug)

///|
pub fn attention_operand_tile_layout(
  rows : Int32,
  columns : Int32,
  side : Int32,
) -> AttentionOperandTileLayout? {
  if rows <= 0 ||
    columns <= 0 ||
    side <= 0 ||
    rows > 65536 ||
    columns > 65536 ||
    rows > 2147483647 / columns ||
    rows % side != 0 ||
    columns % side != 0 {
    None
  } else {
    Some({ rows, columns, side, })
  }
}

///|
pub fn AttentionOperandTileLayout::microtile_side(
  self : AttentionOperandTileLayout,
) -> Int32 {
  self.side
}

///|
pub fn AttentionOperandTileLayout::offset(
  self : AttentionOperandTileLayout,
  row : Int32,
  column : Int32,
) -> Int32? {
  if row < 0 || row >= self.rows || column < 0 || column >= self.columns {
    None
  } else {
    Some(
      (row / self.side * (self.columns / self.side) + column / self.side) *
      self.side *
      self.side +
      row % self.side * self.side +
      column % self.side,
    )
  }
}
