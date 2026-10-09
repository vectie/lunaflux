///|
pub fn moe_decoder_residual_add(
  residual : ArrayView[Double],
  branch : ArrayView[Double],
  row_count~ : Int,
  hidden_width~ : Int,
  limits~ : GlmHybridReferenceLimits,
) -> GlmResidualCompositionResult raise GlmHybridReferenceError {
  limits.check_sequence(row_count)
  limits.check_channels(hidden_width)
  if residual.length() != row_count * hidden_width ||
    branch.length() != residual.length() {
    raise InvalidShape
  }
  require_finite(residual)
  require_finite(branch)
  let output = Array::make(residual.length(), 0.0)
  for index in 0..<output.length() {
    output[index] = residual[index] + branch[index]
  }
  require_finite(output)
  {
    row_count,
    stream_count: 1,
    hidden_width,
    values: ReadOnlyArray::from_array(output),
  }
}

///|
/// Scalar semantic oracle for `post * branch + combine^T * residual`.
/// Candidate source additionally authenticates the explicit F32-to-BF16
/// control and stage round points used by the official decoder implementation.
pub fn hyper_connection_residual_mix(
  post : ArrayView[Double],
  combine : ArrayView[Double],
  residual : ArrayView[Double],
  branch : ArrayView[Double],
  row_count~ : Int,
  stream_count~ : Int,
  hidden_width~ : Int,
  limits~ : GlmHybridReferenceLimits,
) -> GlmResidualCompositionResult raise GlmHybridReferenceError {
  limits.check_sequence(row_count)
  limits.check_channels(hidden_width)
  if stream_count <= 0 ||
    stream_count > limits.maximum_head_count ||
    post.length() != row_count * stream_count ||
    combine.length() != row_count * stream_count * stream_count ||
    residual.length() != row_count * stream_count * hidden_width ||
    branch.length() != row_count * hidden_width {
    raise InvalidShape
  }
  require_finite(post)
  require_finite(combine)
  require_finite(residual)
  require_finite(branch)
  let output = Array::make(residual.length(), 0.0)
  for row in 0..<row_count {
    for destination in 0..<stream_count {
      for component in 0..<hidden_width {
        let mut value = post[row * stream_count + destination] *
          branch[row * hidden_width + component]
        for source in 0..<stream_count {
          value += combine[(row * stream_count + source) * stream_count +
            destination] *
            residual[(row * stream_count + source) * hidden_width + component]
        }
        output[(row * stream_count + destination) * hidden_width + component] = value
      }
    }
  }
  require_finite(output)
  {
    row_count,
    stream_count,
    hidden_width,
    values: ReadOnlyArray::from_array(output),
  }
}

///|
pub fn GlmResidualCompositionResult::row_count(self : Self) -> Int {
  self.row_count
}

///|
pub fn GlmResidualCompositionResult::stream_count(self : Self) -> Int {
  self.stream_count
}

///|
pub fn GlmResidualCompositionResult::hidden_width(self : Self) -> Int {
  self.hidden_width
}

///|
pub fn GlmResidualCompositionResult::values(
  self : Self,
) -> ReadOnlyArray[Double] {
  self.values
}
