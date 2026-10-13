///|
pub suberror PrefillPlanError {
  InvalidGeometry
}

///|
/// Immutable contiguous row tiling. History is the logical prefix, never a
/// padded physical row count. Stateful operators retain sequence order inside
/// each chunk; only its final logical row may publish a sampled token.
pub struct ContiguousPrefillPlan {
  priv tokens : Int
  priv capacity : Int
  priv chunks : Int
}

///|
pub fn ContiguousPrefillPlan::new(
  input_tokens~ : Int,
  row_capacity~ : Int,
  context_capacity~ : Int,
) -> Self raise PrefillPlanError {
  guard input_tokens > 0 &&
    input_tokens < context_capacity &&
    row_capacity > 0 &&
    row_capacity <= context_capacity else {
    raise InvalidGeometry
  }
  {
    tokens: input_tokens,
    capacity: row_capacity,
    chunks: 1 + (input_tokens - 1) / row_capacity,
  }
}

///|
pub fn ContiguousPrefillPlan::chunk_count(self : Self) -> Int {
  self.chunks
}

///|
/// Returns logical [begin,end), not rounded launch dimensions.
pub fn ContiguousPrefillPlan::range(
  self : Self,
  index : Int,
) -> (Int, Int) raise PrefillPlanError {
  guard index >= 0 && index < self.chunks else { raise InvalidGeometry }
  let begin = index * self.capacity
  (begin, begin + (self.tokens - begin).min(self.capacity))
}

///|
pub fn ContiguousPrefillPlan::produces_token(
  self : Self,
  index : Int,
) -> Bool raise PrefillPlanError {
  ignore(self.range(index))
  index == self.chunks - 1
}
