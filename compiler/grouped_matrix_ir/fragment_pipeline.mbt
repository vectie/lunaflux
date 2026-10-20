///|
/// A physical schedule for an unchanged ascending matrix fold. Independent
/// output fragments reuse operands; there are no partial reduction outputs.
/// Publication, register read-ahead and reader retirement are separate edges.
pub struct MatrixFragmentPipeline {
  priv matrix : MatrixTilePlan
  priv row_tile : Int
}

///|
pub fn MatrixFragmentPipeline::new(
  matrix : MatrixTilePlan,
  row_tile~ : Int,
) -> Self raise @precision.PrecisionPlanError {
  let (_, _, inner) = matrix.geometry()
  guard row_tile is (32 | 64) && inner % 64 == 0 else {
    raise @precision.PrecisionPlanError::NoImplementation
  }
  { matrix, row_tile, }
}

///|
pub fn MatrixFragmentPipeline::matrix(self : Self) -> MatrixTilePlan {
  self.matrix
}

///|
pub fn MatrixFragmentPipeline::tile(self : Self) -> (Int, Int, Int) {
  (self.row_tile, 64, 64)
}

///|
pub fn MatrixFragmentPipeline::operand_slots(_self : Self) -> Int {
  2
}

///|
pub fn MatrixFragmentPipeline::fragment_slots(_self : Self) -> Int {
  2
}

///|
pub fn MatrixFragmentPipeline::operand_alignment(_self : Self) -> Int64 {
  16L
}

///|
pub fn MatrixFragmentPipeline::shared_bytes(self : Self) -> Int {
  self.operand_slots() * (self.row_tile + 64) * 64 * 2
}

///|
pub fn MatrixFragmentPipeline::tile_count(self : Self) -> Int {
  let (rows, columns, _) = self.matrix.geometry()
  let matrix = (rows + self.row_tile - 1) /
    self.row_tile *
    ((columns + 63) / 64)
  if self.matrix.law() == ShapeAwareFloat32 {
    Int::max(matrix, (columns + 3) / 4)
  } else {
    matrix
  }
}

///|
/// Bijective byte permutation within a 1024-byte operand slab. Both producer
/// and matrix reader consume this same map; 16-byte copy atoms stay contiguous.
pub fn MatrixFragmentPipeline::operand_byte_offset(
  _self : Self,
  logical : Int,
) -> Int {
  logical ^ ((logical >> 3) & 112)
}
