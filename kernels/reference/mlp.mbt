///|
/// Computes the Llama gated MLP:
/// `down(silu(gate(input)) * up(input))`.
pub fn silu_gated_mlp(
  input : Activation,
  gate_weight : Tensor,
  up_weight : Tensor,
  down_weight : Tensor,
  limits~ : ReferenceLimits,
) -> Activation raise ReferenceKernelError {
  if gate_weight.columns != input.columns {
    raise InvalidShape(MlpGateWeight)
  }
  if up_weight.columns != input.columns || up_weight.rows != gate_weight.rows {
    raise InvalidShape(MlpUpWeight)
  }
  if down_weight.columns != gate_weight.rows {
    raise InvalidShape(MlpDownWeight)
  }
  let intermediate_elements = checked_element_count(
    input.rows,
    gate_weight.rows,
    limits,
  )
  let gate = project_values(input, gate_weight, limits)
  let up = project_values(input, up_weight, limits)
  let combined = Array::make(intermediate_elements, 0.0F)
  for index in 0..<intermediate_elements {
    let gate_value = gate[index]
    combined[index] = gate_value / (1.0F + @math.expf(-gate_value)) * up[index]
  }
  let intermediate = activation_from_owned(
    input.rows,
    gate_weight.rows,
    combined,
    limits,
  )
  dense_projection(intermediate, down_weight, limits~)
}
