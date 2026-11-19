///|
/// Bit-preserving U16 row insertion, indexed by causal prompt positions.
/// Feature lookup is separate from rotary coordinates and token vocabulary.
pub struct EmbeddingScatterProgram {
  priv width : Int32
  priv rows : Int32
  priv positions : Int32
  priv features : Int32
  priv ownership : @physical_tile_ir.VectorOwnership
}

///|
pub fn EmbeddingScatterProgram::new(
  width~ : Int32,
  maximum_rows~ : Int32,
  maximum_positions~ : Int32,
  maximum_features~ : Int32,
  owners~ : Int32,
) -> Self? {
  guard width > 0 &&
    maximum_rows > 0 &&
    maximum_positions >= maximum_rows &&
    maximum_features > 0 &&
    maximum_features <= maximum_positions &&
    width.to_int64() * maximum_rows.to_int64() <= 2147483647L &&
    width.to_int64() * maximum_features.to_int64() <= 2147483647L &&
    @physical_tile_ir.VectorOwnership::new(vectors=width, owners~)
    is Some(ownership) else {
    return None
  }
  Some({
    width,
    rows: maximum_rows,
    positions: maximum_positions,
    features: maximum_features,
    ownership,
  })
}

///|
pub fn EmbeddingScatterProgram::geometry(
  self : Self,
) -> (Int32, Int32, Int32, Int32) {
  (self.width, self.rows, self.positions, self.features)
}

///|
pub fn EmbeddingScatterProgram::ownership(
  self : Self,
) -> @physical_tile_ir.VectorOwnership {
  self.ownership
}
