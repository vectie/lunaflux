///|
fn key_value_projection_roles() -> ReadOnlyArray[
  AdvancedDecoderLaunchOperandRole,
] {
  [
    StepCountsInput,
    HiddenInput,
    KeyValueWeightCodesInput,
    KeyValueWeightScalesInput,
    KeyValueProjectionOutput,
  ]
}

///|
fn key_value_projection_operands_are_exact(
  heads : Int,
  head_dimension : Int,
  rope_dimension : Int,
  operands : ReadOnlyArray[AdvancedDecoderLaunchOperand],
  dimensions : @launch_contract.AotLaunchDimensions,
  geometry : @advanced_decoder_kernel_requirements.AdvancedDecoderKernelGeometry,
) -> Bool {
  let hidden = geometry.hidden_size()
  if heads != 1 ||
    head_dimension <= 0 ||
    rope_dimension <= 0 ||
    rope_dimension > head_dimension ||
    hidden <= 0 ||
    head_dimension % 128 != 0 ||
    hidden % 128 != 0 ||
    dimensions.grid_x() <= 0 ||
    dimensions.grid_y() != (head_dimension + 255) / 256 ||
    dimensions.grid_z() != 1 ||
    dimensions.block_x() != 256 ||
    dimensions.block_y() != 1 ||
    dimensions.block_z() != 1 ||
    dimensions.shared_memory_bytes() != 0 {
    return false
  }
  let output_width = head_dimension
  let rows = dimensions.grid_x().to_int64()
  let input_width = hidden.to_int64()
  let output = output_width.to_int64()
  if rows > 9223372036854775807L / input_width ||
    rows * input_width > 9223372036854775807L / 2L ||
    output > 9223372036854775807L / input_width ||
    rows > 9223372036854775807L / output ||
    rows * output > 9223372036854775807L / 2L {
    return false
  }
  let input_bytes = rows * input_width * 2L
  let weight_bytes = output * input_width
  let scale_bytes = output / 128L * (input_width / 128L)
  let output_bytes = rows * output * 2L
  operands[0].byte_count() == 20L &&
  operands[0].alignment() == 4L &&
  operands[1].byte_count() == input_bytes &&
  operands[1].alignment() == 16L &&
  operands[2].byte_count() == weight_bytes &&
  operands[2].alignment() == 16L &&
  operands[3].byte_count() == scale_bytes &&
  operands[3].alignment() == 16L &&
  operands[4].byte_count() == output_bytes &&
  operands[4].alignment() == 16L
}
