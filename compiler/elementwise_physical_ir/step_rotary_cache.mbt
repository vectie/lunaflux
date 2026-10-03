///|
/// A bounded immutable rotary operand table produced once for each submitted
/// token domain, then borrowed by every compatible layer. Preparation and all
/// consumers belong to the same ordered execution graph; retirement occurs
/// only after its final consumer, before the next step may overwrite cells.
pub struct StepRotaryCachePlan {
  priv maximum_tokens : Int
  priv pairs : Int
} derive(Eq, Debug)

///|
pub fn StepRotaryCachePlan::new(
  maximum_tokens~ : Int,
  head_dimension~ : Int,
) -> StepRotaryCachePlan? {
  guard maximum_tokens > 0 &&
    maximum_tokens <= 1048576 &&
    head_dimension > 0 &&
    head_dimension <= 65536 &&
    head_dimension % 2 == 0 else {
    return None
  }
  Some({ maximum_tokens, pairs: head_dimension / 2, })
}

///|
/// (maximum token cells, rotary pairs per token). Each pair stores sine then
/// cosine as two F32 values. Position is a step input, not a cache key.
pub fn StepRotaryCachePlan::geometry(self : StepRotaryCachePlan) -> (Int, Int) {
  (self.maximum_tokens, self.pairs)
}

///|
pub fn StepRotaryCachePlan::byte_count(self : StepRotaryCachePlan) -> Int64 {
  self.maximum_tokens.to_int64() * self.pairs.to_int64() * 8L
}

///|
pub fn StepRotaryCachePlan::cell_offset(
  self : StepRotaryCachePlan,
  token : Int,
  pair : Int,
) -> Int64? {
  guard token >= 0 &&
    token < self.maximum_tokens &&
    pair >= 0 &&
    pair < self.pairs else {
    return None
  }
  Some((token.to_int64() * self.pairs.to_int64() + pair.to_int64()) * 8L)
}
