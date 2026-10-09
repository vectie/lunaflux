///|
/// Tiny scalar row-major no-bias dense projection matching the terminal KDA
/// output projection. BF16 storage and round points belong to the CUDA source
/// contract; this oracle authenticates shape and accumulation ordering.
pub fn kda_output_projection(
  input : ArrayView[Double],
  weights : ArrayView[Double],
  row_count~ : Int,
  input_width~ : Int,
  output_width~ : Int,
  limits~ : GlmHybridReferenceLimits,
) -> GlmKdaOutputProjectionResult raise GlmHybridReferenceError {
  limits.check_sequence(row_count)
  limits.check_channels(input_width)
  limits.check_channels(output_width)
  if input.length() != row_count * input_width ||
    weights.length() != output_width * input_width {
    raise InvalidShape
  }
  require_finite(input)
  require_finite(weights)
  let values = Array::make(row_count * output_width, 0.0)
  for row in 0..<row_count {
    let input_base = row * input_width
    let output_base = row * output_width
    for output in 0..<output_width {
      values[output_base + output] = kda_dense_row(
        input, weights, input_base, output, input_width,
      )
    }
  }
  {
    row_count,
    input_width,
    output_width,
    values: ReadOnlyArray::from_array(values),
  }
}

///|
pub fn GlmKdaOutputProjectionResult::row_count(self : Self) -> Int {
  self.row_count
}

///|
pub fn GlmKdaOutputProjectionResult::input_width(self : Self) -> Int {
  self.input_width
}

///|
pub fn GlmKdaOutputProjectionResult::output_width(self : Self) -> Int {
  self.output_width
}

///|
pub fn GlmKdaOutputProjectionResult::values(
  self : Self,
) -> ReadOnlyArray[Double] {
  self.values
}
