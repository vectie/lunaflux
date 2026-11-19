///|
pub(all) enum LinearProjection {
  QueryKeyValue
  Gate
  Decay
  Beta
  Output
}

///|
/// BF16 projections around the recurrent branch. Q/K/V retain the checkpoint's
/// packed row order; gate and controls have independent matrix outputs.
pub fn DecoderOperators::linear_projection(
  self : Self,
  site : LinearProjection,
) -> @matrix.MatrixTilePlan raise {
  let geometry = self.spec.decoder()
  let linear = geometry.recurrent_attention()
  let value = linear.value_heads * linear.value_width
  let qk = linear.key_heads * linear.key_width
  let (columns, inner) = match site {
    QueryKeyValue => (qk * 2 + value, geometry.hidden_width())
    Gate => (value, geometry.hidden_width())
    Decay | Beta => (linear.value_heads, geometry.hidden_width())
    Output => (geometry.hidden_width(), value)
  }
  @matrix.MatrixTilePlan::new(
    rows=self.maximum_rows,
    columns~,
    inner~,
    law=ShapeAwareFloat32,
  )
}
