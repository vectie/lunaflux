///|
/// A bijective physical permutation of a complete logical plane. The inner
/// layout defines shared producer/consumer addresses; outer tiles are stored
/// row-major. No values, scales or numerical operations change.
pub struct TiledPlane {
  priv rows : Int
  priv columns : Int
  priv tile_rows : Int
  priv layout : OperandLayout
} derive(Eq, Debug)

///|
pub fn TiledPlane::new(
  rows~ : Int,
  columns~ : Int,
  tile_rows~ : Int,
  layout~ : OperandLayout,
) -> Self? {
  let (width, _, rotations, row_group) = layout.geometry()
  guard rows > 0 &&
    columns > 0 &&
    tile_rows > 0 &&
    rows % tile_rows == 0 &&
    columns % width == 0 &&
    layout.row_stride() == width &&
    tile_rows.to_int64() % (rotations.to_int64() * row_group.to_int64()) == 0L &&
    rows.to_int64() * columns.to_int64() <= 2147483647L else {
    return None
  }
  Some({ rows, columns, tile_rows, layout, })
}

///|
pub fn TiledPlane::geometry(self : Self) -> (Int, Int, Int, Int) {
  (self.rows, self.columns, self.tile_rows, self.layout.geometry().0)
}

///|
pub fn TiledPlane::offset(self : Self, row : Int, column : Int) -> Int? {
  guard row >= 0 && row < self.rows && column >= 0 && column < self.columns else {
    return None
  }
  let width = self.layout.geometry().0
  let tile = row / self.tile_rows * (self.columns / width) + column / width
  Some(
    tile * self.tile_rows * width +
    self.layout.offset(row % self.tile_rows, column % width).unwrap(),
  )
}

///|
pub fn TiledPlane::elements(self : Self) -> Int {
  self.rows * self.columns
}

///|
pub fn TiledPlane::layout(self : Self) -> OperandLayout {
  self.layout
}
