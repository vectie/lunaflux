///|
fn kda_dense_row(
  input : ArrayView[Double],
  weights : ArrayView[Double],
  input_base : Int,
  output_index : Int,
  input_width : Int,
) -> Double {
  let mut sum = 0.0
  let weight_base = output_index * input_width
  for input_index in 0..<input_width {
    sum += input[input_base + input_index] * weights[weight_base + input_index]
  }
  sum
}

///|
/// Tiny scalar semantics for the BF16-side recurrent-attention projection
/// stage. The CUDA contract supplies the explicit BF16 round points; this
/// reference authenticates tensor ordering, the two-stage forget projection,
/// and input-gate sigmoid without absorbing F32 decay control.
pub fn kda_projection_control(
  hidden : ArrayView[Double],
  query_weights : ArrayView[Double],
  key_weights : ArrayView[Double],
  value_weights : ArrayView[Double],
  forget_a_weights : ArrayView[Double],
  forget_b_weights : ArrayView[Double],
  input_gate_weights : ArrayView[Double],
  row_count~ : Int,
  hidden_width~ : Int,
  head_count~ : Int,
  head_dimension~ : Int,
  control_rank~ : Int,
  limits~ : GlmHybridReferenceLimits,
) -> GlmKdaProjectionResult raise GlmHybridReferenceError {
  limits.check_sequence(row_count)
  limits.check_heads(head_count, head_dimension)
  limits.check_channels(hidden_width)
  limits.check_channels(control_rank)
  let projected_width = head_count * head_dimension
  limits.check_channels(projected_width)
  let hidden_cells = row_count * hidden_width
  let projected_cells = row_count * projected_width
  let low_cells = row_count * control_rank
  let gate_cells = row_count * head_count
  if hidden.length() != hidden_cells ||
    query_weights.length() != projected_width * hidden_width ||
    key_weights.length() != projected_width * hidden_width ||
    value_weights.length() != projected_width * hidden_width ||
    forget_a_weights.length() != control_rank * hidden_width ||
    forget_b_weights.length() != projected_width * control_rank ||
    input_gate_weights.length() != head_count * hidden_width {
    raise InvalidShape
  }
  for
    values in [
      hidden, query_weights, key_weights, value_weights, forget_a_weights, forget_b_weights,
      input_gate_weights,
    ] {
    require_finite(values)
  }
  let query = Array::make(projected_cells, 0.0)
  let key = Array::make(projected_cells, 0.0)
  let value = Array::make(projected_cells, 0.0)
  let forget_low_rank = Array::make(low_cells, 0.0)
  let raw_forget = Array::make(projected_cells, 0.0)
  let beta = Array::make(gate_cells, 0.0)
  for row in 0..<row_count {
    let hidden_base = row * hidden_width
    let projected_base = row * projected_width
    for output in 0..<projected_width {
      query[projected_base + output] = kda_dense_row(
        hidden, query_weights, hidden_base, output, hidden_width,
      )
      key[projected_base + output] = kda_dense_row(
        hidden, key_weights, hidden_base, output, hidden_width,
      )
      value[projected_base + output] = kda_dense_row(
        hidden, value_weights, hidden_base, output, hidden_width,
      )
    }
    let low_base = row * control_rank
    for output in 0..<control_rank {
      forget_low_rank[low_base + output] = kda_dense_row(
        hidden, forget_a_weights, hidden_base, output, hidden_width,
      )
    }
    for output in 0..<projected_width {
      raw_forget[projected_base + output] = kda_dense_row(
        forget_low_rank, forget_b_weights, low_base, output, control_rank,
      )
    }
    let gate_base = row * head_count
    for output in 0..<head_count {
      let logit = kda_dense_row(
        hidden, input_gate_weights, hidden_base, output, hidden_width,
      )
      beta[gate_base + output] = 1.0 / (1.0 + @math.exp(-logit))
    }
  }
  {
    row_count,
    projected_width,
    control_rank,
    head_count,
    query: ReadOnlyArray::from_array(query),
    key: ReadOnlyArray::from_array(key),
    value: ReadOnlyArray::from_array(value),
    forget_low_rank: ReadOnlyArray::from_array(forget_low_rank),
    raw_forget: ReadOnlyArray::from_array(raw_forget),
    beta: ReadOnlyArray::from_array(beta),
  }
}
