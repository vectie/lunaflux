///|
pub(all) enum FullProjection {
  QueryAndGate
  Key
  Value
  Output
}

///|
/// Startup-only matrix plan for BF16 full-attention projections. Q retains its
/// [head,query-or-gate,channel] checkpoint layout; the output consumes all heads.
pub fn DecoderOperators::full_projection(
  self : Self,
  site : FullProjection,
) -> @matrix.MatrixTilePlan raise {
  let geometry = self.spec.decoder()
  let full = geometry.full_attention()
  let query = full.query_heads * full.head_width
  let kv = full.kv_heads * full.head_width
  let (columns, inner) = match site {
    QueryAndGate => (query * 2, geometry.hidden_width())
    Key | Value => (kv, geometry.hidden_width())
    Output => (geometry.hidden_width(), query)
  }
  @matrix.MatrixTilePlan::new(
    rows=self.maximum_rows,
    columns~,
    inner~,
    law=ShapeAwareFloat32,
  )
}
