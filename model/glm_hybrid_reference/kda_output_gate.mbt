///|
pub fn kda_output_gate_projection(
  hidden : ArrayView[Double],
  first_weights : ArrayView[Double],
  second_weights : ArrayView[Double],
  row_count~ : Int,
  hidden_width~ : Int,
  head_count~ : Int,
  head_dimension~ : Int,
  limits~ : GlmHybridReferenceLimits,
) -> GlmKdaOutputGateResult raise GlmHybridReferenceError {
  limits.check_sequence(row_count)
  limits.check_heads(head_count, head_dimension)
  limits.check_channels(hidden_width)
  let output_width = head_count * head_dimension
  limits.check_channels(output_width)
  if hidden.length() != row_count * hidden_width ||
    first_weights.length() != head_dimension * hidden_width ||
    second_weights.length() != output_width * head_dimension {
    raise InvalidShape
  }
  require_finite(hidden)
  require_finite(first_weights)
  require_finite(second_weights)
  let low_rank = Array::make(row_count * head_dimension, 0.0)
  let gate = Array::make(row_count * output_width, 0.0)
  for row in 0..<row_count {
    let hidden_base = row * hidden_width
    let low_base = row * head_dimension
    for output in 0..<head_dimension {
      low_rank[low_base + output] = kda_dense_row(
        hidden, first_weights, hidden_base, output, hidden_width,
      )
    }
    let gate_base = row * output_width
    for output in 0..<output_width {
      gate[gate_base + output] = kda_dense_row(
        low_rank, second_weights, low_base, output, head_dimension,
      )
    }
  }
  {
    low_rank: ReadOnlyArray::from_array(low_rank),
    gate: ReadOnlyArray::from_array(gate),
  }
}

///|
pub fn GlmKdaOutputGateResult::low_rank(self : Self) -> ReadOnlyArray[Double] {
  self.low_rank
}

///|
pub fn GlmKdaOutputGateResult::gate(self : Self) -> ReadOnlyArray[Double] {
  self.gate
}
