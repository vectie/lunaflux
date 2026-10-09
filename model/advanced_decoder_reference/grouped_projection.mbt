///|
/// Correctness-only oracle for row-major group-local projection. Input is
/// `[row,group,input]`, weights are `[group,output,input]`, and output is
/// `[row,group,output]`. Arithmetic is ordered F32; callers model the final
/// BF16 boundary separately.
pub fn grouped_output_a_projection_v1(
  input : ArrayView[Float],
  weights : ArrayView[Float],
  row_count~ : Int,
  group_count~ : Int,
  input_width~ : Int,
  output_width~ : Int,
  limits~ : ReferenceLimits,
) -> ReadOnlyArray[Float] raise AdvancedDecoderReferenceError {
  if row_count <= 0 ||
    group_count <= 0 ||
    input_width <= 0 ||
    output_width <= 0 ||
    group_count > limits.max_width() ||
    input_width > limits.max_width() ||
    output_width > limits.max_width() ||
    input.length() != row_count * group_count * input_width ||
    weights.length() != group_count * output_width * input_width ||
    row_count * group_count * output_width > limits.max_values() {
    raise InvalidDimension
  }
  validate_values(input, limits)
  validate_values(weights, limits)
  let output = Array::make(row_count * group_count * output_width, 0.0F)
  for row in 0..<row_count {
    for group in 0..<group_count {
      for column in 0..<output_width {
        let mut accumulator = 0.0F
        for inner in 0..<input_width {
          let input_index = (row * group_count + group) * input_width + inner
          let weight_index = (group * output_width + column) * input_width +
            inner
          accumulator = accumulator + input[input_index] * weights[weight_index]
        }
        output[(row * group_count + group) * output_width + column] = accumulator
      }
    }
  }
  ReadOnlyArray::from_array(output)
}
