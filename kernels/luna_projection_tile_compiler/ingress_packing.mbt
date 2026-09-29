///|
/// Lifetime of row-invariant rotary values. No head may overwrite the values
/// while another complete-head epilogue still consumes them.
pub(all) enum AttentionIngressReuseEffect {
  PrepareRowRotary
  ConsumeCompleteHeads
  RetireRowRotary
} derive(Debug, Eq)

///|
/// Pure ownership refinement for a product of token rows and packed Q/K/V
/// heads. Projection shares each input tile across the local head domain.
/// Rotary sine/cosine pairs are row-scoped and reused by local Q/K heads;
/// values never observe them. A tail has no output or KV-store authority.
pub struct AttentionIngressPacking {
  priv query_heads : Int
  priv kv_heads : Int
  priv head_dimension : Int
  priv heads_per_group : Int
  priv lanes : Int
} derive(Debug, Eq)

///|
pub fn ProjectionAttentionIngressEpilogue::packing(
  self : ProjectionAttentionIngressEpilogue,
  heads_per_group~ : Int,
  reduction_lanes~ : Int,
) -> AttentionIngressPacking raise ProjectionTileCompilerError {
  ignore(self.numerics(reduction_lanes~))
  let heads = self.query_heads.to_int64() + 2L * self.kv_heads.to_int64()
  guard heads > 0L &&
    heads <= 2147483647L / self.head_dimension.to_int64() &&
    heads_per_group > 0 &&
    heads_per_group <= 2147483647 / self.head_dimension else {
    raise InvalidProjectionTileCompilation(InvalidProblem)
  }
  {
    query_heads: self.query_heads,
    kv_heads: self.kv_heads,
    head_dimension: self.head_dimension,
    heads_per_group,
    lanes: reduction_lanes,
  }
}

///|
/// (workgroups, heads per group, producer columns, rotary pairs per owner).
pub fn AttentionIngressPacking::geometry(
  self : AttentionIngressPacking,
) -> (Int, Int, Int, Int) {
  let heads = self.query_heads + 2 * self.kv_heads
  (
    (heads - 1) / self.heads_per_group + 1,
    self.heads_per_group,
    self.heads_per_group * self.head_dimension,
    (self.head_dimension / 2 - 1) / self.lanes + 1,
  )
}

///|
/// Resolve a producer column to (Q/K/V operand ordinal, operand-local row).
/// Returning None means masked zero-fill, not a borrowed adjacent matrix.
pub fn AttentionIngressPacking::weight_row(
  self : AttentionIngressPacking,
  workgroup : Int,
  column : Int,
) -> (Int, Int)? {
  let (groups, _, columns, _) = self.geometry()
  guard workgroup >= 0 && workgroup < groups && column >= 0 && column < columns else {
    return None
  }
  let head = workgroup.to_int64() * self.heads_per_group.to_int64() +
    (column / self.head_dimension).to_int64()
  let q = self.query_heads.to_int64()
  let k = self.kv_heads.to_int64()
  let component = column % self.head_dimension
  if head < q {
    Some((0, head.to_int() * self.head_dimension + component))
  } else if head < q + k {
    Some((1, (head - q).to_int() * self.head_dimension + component))
  } else if head < q + 2L * k {
    Some((2, (head - q - k).to_int() * self.head_dimension + component))
  } else {
    None
  }
}

///|
pub fn AttentionIngressPacking::effects(
  _self : AttentionIngressPacking,
) -> ReadOnlyArray[AttentionIngressReuseEffect] {
  [PrepareRowRotary, ConsumeCompleteHeads, RetireRowRotary]
}

///|
pub fn AttentionIngressPacking::canonical_bytes(
  self : AttentionIngressPacking,
) -> Bytes {
  @utf8.encode(
    "luna-attention-ingress-packing-v1\n" +
    "heads=\{self.query_heads},\{self.kv_heads},\{self.head_dimension}\n" +
    "heads_per_group=\{self.heads_per_group}\n" +
    "reduction_lanes=\{self.lanes}\n" +
    "input_scope=reduction-tile\nrotary_scope=token-row\n" +
    "effects=prepare-row-rotary,consume-complete-heads,retire-row-rotary\n" +
    "tail=zero-fill-no-store\nnumerics=unchanged-per-head\n",
  )
}
