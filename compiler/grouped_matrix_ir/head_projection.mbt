///|
/// Independent row/head matrices with a strided shared weight bank. The query
/// head maps to a weight head by a complete integral group. Transposition is
/// expressed by strides, not backend or model-family branching.
pub struct HeadProjectionPlan {
  priv tile : MatrixTilePlan
  priv heads : Int
  priv weight_heads : Int
  priv head_stride : Int64
  priv column_stride : Int64
  priv inner_stride : Int64
  priv offset : Int64
}

///|
pub fn HeadProjectionPlan::new(
  rows~ : Int,
  heads~ : Int,
  weight_heads~ : Int,
  columns~ : Int,
  inner~ : Int,
  head_stride~ : Int64,
  column_stride~ : Int64,
  inner_stride~ : Int64,
  offset~ : Int64,
  weight_elements~ : Int64,
) -> Self raise @precision.PrecisionPlanError {
  guard heads > 0 &&
    heads <= 1024 &&
    weight_heads > 0 &&
    heads % weight_heads == 0 &&
    columns > 0 &&
    columns <= 65536 &&
    inner > 0 &&
    inner <= 65536 &&
    [head_stride, column_stride, inner_stride].all(s => {
      s > 0L && s <= 4294967296L
    }) &&
    offset >= 0L &&
    offset <= 4294967296L else {
    raise @precision.PrecisionPlanError::IllegalPlacement
  }
  let end = offset +
    (weight_heads - 1).to_int64() * head_stride +
    (columns - 1).to_int64() * column_stride +
    (inner - 1).to_int64() * inner_stride +
    1L
  guard end <= weight_elements else {
    raise @precision.PrecisionPlanError::IllegalPlacement
  }
  let tile = MatrixTilePlan::new(rows~, columns~, inner~, law=MatrixFloat32)
  guard tile.tile_count().to_int64() * heads.to_int64() <= 2147483647L else {
    raise @precision.PrecisionPlanError::ArithmeticOverflow
  }
  {
    tile,
    heads,
    weight_heads,
    head_stride,
    column_stride,
    inner_stride,
    offset,
  }
}

///|
pub fn HeadProjectionPlan::tile(self : Self) -> MatrixTilePlan {
  self.tile
}

///|
pub fn HeadProjectionPlan::heads(self : Self) -> (Int, Int) {
  (self.heads, self.weight_heads)
}

///|
pub fn HeadProjectionPlan::strides(self : Self) -> (Int64, Int64, Int64, Int64) {
  (self.head_stride, self.column_stride, self.inner_stride, self.offset)
}

///|
pub fn HeadProjectionPlan::workgroups(self : Self) -> Int {
  self.tile.tile_count() * self.heads
}
