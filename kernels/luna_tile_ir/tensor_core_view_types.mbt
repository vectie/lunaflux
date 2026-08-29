///|
/// Opaque proof that one admitted program has the exact BF16 row-major
/// 16x16x16 MMA graph accepted by the first tensor-core candidate boundary.
pub struct LunaTileBf16Mma16x16x16View {
  priv program_digest : LunaTileProgramDigest
  priv tile_count : Int
  priv left_global : LunaTileTensorId
  priv right_global : LunaTileTensorId
  priv accumulator_global : LunaTileTensorId
  priv output_global : LunaTileTensorId
}

///|
pub fn LunaTileBf16Mma16x16x16View::program_digest(
  self : LunaTileBf16Mma16x16x16View,
) -> LunaTileProgramDigest {
  self.program_digest
}

///|
pub fn LunaTileBf16Mma16x16x16View::tile_count(
  self : LunaTileBf16Mma16x16x16View,
) -> Int {
  self.tile_count
}

///|
pub fn LunaTileBf16Mma16x16x16View::left_global(
  self : LunaTileBf16Mma16x16x16View,
) -> LunaTileTensorId {
  self.left_global
}

///|
pub fn LunaTileBf16Mma16x16x16View::right_global(
  self : LunaTileBf16Mma16x16x16View,
) -> LunaTileTensorId {
  self.right_global
}

///|
pub fn LunaTileBf16Mma16x16x16View::accumulator_global(
  self : LunaTileBf16Mma16x16x16View,
) -> LunaTileTensorId {
  self.accumulator_global
}

///|
pub fn LunaTileBf16Mma16x16x16View::output_global(
  self : LunaTileBf16Mma16x16x16View,
) -> LunaTileTensorId {
  self.output_global
}
