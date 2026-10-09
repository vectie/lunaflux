///|
fn moe_expert_row(
  input : ArrayView[Double],
  expert : Int,
  hidden_width : Int,
  intermediate_width : Int,
  gate_weights : ArrayView[Double],
  up_weights : ArrayView[Double],
  down_weights : ArrayView[Double],
  activation : @advanced_decoder_execution_plan.ExpertActivation,
) -> Array[Double] {
  let output = Array::make(hidden_width, 0.0)
  for inner in 0..<intermediate_width {
    let weight_base = (expert * intermediate_width + inner) * hidden_width
    let mut gate = 0.0
    let mut up = 0.0
    for column in 0..<hidden_width {
      gate += input[column] * gate_weights[weight_base + column]
      up += input[column] * up_weights[weight_base + column]
    }
    match activation {
      SwiGlu => ()
      ClampedSwiGlu(limit) => {
        gate = gate.min(limit)
        up = up.max(-limit).min(limit)
      }
    }
    let product = gate / (1.0 + @math.exp(-gate)) * up
    let down_base = expert * hidden_width * intermediate_width
    for column in 0..<hidden_width {
      output[column] += product *
        down_weights[down_base + column * intermediate_width + inner]
    }
  }
  output
}

///|
/// Tiny selected-expert oracle. It validates indices before evaluating experts
/// in ascending expert-id order, matching the correctness source contract.
pub fn moe_routed_expert_feed_forward(
  input : ArrayView[Double],
  selected_indices : ArrayView[Int],
  selected_weights : ArrayView[Float],
  gate_weights : ArrayView[Double],
  up_weights : ArrayView[Double],
  down_weights : ArrayView[Double],
  row_count~ : Int,
  hidden_width~ : Int,
  intermediate_width~ : Int,
  expert_count~ : Int,
  experts_per_token~ : Int,
  activation~ : @advanced_decoder_execution_plan.ExpertActivation,
  limits~ : GlmHybridReferenceLimits,
) -> GlmMoeFeedForwardResult raise GlmHybridReferenceError {
  limits.check_sequence(row_count)
  limits.check_channels(hidden_width)
  limits.check_channels(intermediate_width)
  if expert_count <= 0 ||
    experts_per_token <= 0 ||
    experts_per_token > expert_count ||
    input.length() != row_count * hidden_width ||
    selected_indices.length() != row_count * experts_per_token ||
    selected_weights.length() != selected_indices.length() ||
    gate_weights.length() != expert_count * intermediate_width * hidden_width ||
    up_weights.length() != gate_weights.length() ||
    down_weights.length() != expert_count * hidden_width * intermediate_width {
    raise InvalidShape
  }
  require_finite(input)
  require_finite(gate_weights)
  require_finite(up_weights)
  require_finite(down_weights)
  let output = Array::make(row_count * hidden_width, 0.0)
  for row in 0..<row_count {
    let selected_base = row * experts_per_token
    for slot in 0..<experts_per_token {
      let expert = selected_indices[selected_base + slot]
      if expert < 0 || expert >= expert_count {
        raise IndexOutOfBounds
      }
      if selected_indices[selected_base:selected_base + slot].contains(expert) {
        raise DuplicateIndex
      }
      if !oracle_float_finite(selected_weights[selected_base + slot]) {
        raise NonFiniteInput
      }
    }
    for expert in 0..<expert_count {
      for slot in 0..<experts_per_token {
        if selected_indices[selected_base + slot] == expert {
          let expert_output = moe_expert_row(
            input[row * hidden_width:(row + 1) * hidden_width],
            expert,
            hidden_width,
            intermediate_width,
            gate_weights,
            up_weights,
            down_weights,
            activation,
          )
          let weight = selected_weights[selected_base + slot].to_double()
          for column in 0..<hidden_width {
            output[row * hidden_width + column] += expert_output[column] *
              weight
          }
        }
      }
    }
  }
  require_finite(output)
  { row_count, hidden_width, values: ReadOnlyArray::from_array(output), }
}

///|
pub fn moe_shared_expert_feed_forward(
  input : ArrayView[Double],
  gate_weights : ArrayView[Double],
  up_weights : ArrayView[Double],
  down_weights : ArrayView[Double],
  row_count~ : Int,
  hidden_width~ : Int,
  intermediate_width~ : Int,
  activation~ : @advanced_decoder_execution_plan.ExpertActivation,
  limits~ : GlmHybridReferenceLimits,
) -> GlmMoeFeedForwardResult raise GlmHybridReferenceError {
  moe_routed_expert_feed_forward(
    input,
    Array::make(row_count, 0),
    Array::make(row_count, 1.0F),
    gate_weights,
    up_weights,
    down_weights,
    row_count~,
    hidden_width~,
    intermediate_width~,
    expert_count=1,
    experts_per_token=1,
    activation~,
    limits~,
  )
}

///|
pub fn moe_routed_shared_combine(
  routed : ArrayView[Double],
  shared : ArrayView[Double],
  row_count~ : Int,
  hidden_width~ : Int,
  limits~ : GlmHybridReferenceLimits,
) -> GlmMoeFeedForwardResult raise GlmHybridReferenceError {
  limits.check_sequence(row_count)
  limits.check_channels(hidden_width)
  if routed.length() != row_count * hidden_width ||
    shared.length() != routed.length() {
    raise InvalidShape
  }
  require_finite(routed)
  require_finite(shared)
  let output = Array::make(routed.length(), 0.0)
  for index in 0..<output.length() {
    output[index] = routed[index] + shared[index]
  }
  require_finite(output)
  { row_count, hidden_width, values: ReadOnlyArray::from_array(output), }
}

///|
pub fn GlmMoeFeedForwardResult::row_count(self : Self) -> Int {
  self.row_count
}

///|
pub fn GlmMoeFeedForwardResult::hidden_width(self : Self) -> Int {
  self.hidden_width
}

///|
pub fn GlmMoeFeedForwardResult::values(self : Self) -> ReadOnlyArray[Double] {
  self.values
}
