///|
/// Tiny scalar row-major router projection. The production candidate binds an
/// F32 activation view and BF16 checkpoint weight; score transforms and expert
/// selection are deliberately downstream.
pub fn moe_router_projection(
  input : ArrayView[Double],
  weight : ArrayView[Double],
  row_count~ : Int,
  hidden_width~ : Int,
  expert_count~ : Int,
  limits~ : GlmHybridReferenceLimits,
) -> GlmMoeRouterProjectionResult raise GlmHybridReferenceError {
  limits.check_sequence(row_count)
  limits.check_channels(hidden_width)
  limits.check_channels(expert_count)
  if input.length() != row_count * hidden_width ||
    weight.length() != expert_count * hidden_width {
    raise InvalidShape
  }
  require_finite(input)
  require_finite(weight)
  let logits = Array::make(row_count * expert_count, 0.0)
  for row in 0..<row_count {
    let input_base = row * hidden_width
    let output_base = row * expert_count
    for expert in 0..<expert_count {
      logits[output_base + expert] = kda_dense_row(
        input, weight, input_base, expert, hidden_width,
      )
    }
  }
  {
    row_count,
    hidden_width,
    expert_count,
    logits: ReadOnlyArray::from_array(logits),
  }
}

///|
pub fn GlmMoeRouterProjectionResult::row_count(self : Self) -> Int {
  self.row_count
}

///|
pub fn GlmMoeRouterProjectionResult::hidden_width(self : Self) -> Int {
  self.hidden_width
}

///|
pub fn GlmMoeRouterProjectionResult::expert_count(self : Self) -> Int {
  self.expert_count
}

///|
pub fn GlmMoeRouterProjectionResult::logits(
  self : Self,
) -> ReadOnlyArray[Double] {
  self.logits
}
