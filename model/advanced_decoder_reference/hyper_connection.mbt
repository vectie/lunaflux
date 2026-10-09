///|
/// Computes a bounded standard Sinkhorn normalization of a square logit
/// matrix. A global logit shift is removed before exponentiation for stability.
pub fn sinkhorn_normalize(
  logits : ArrayView[Float],
  width~ : Int,
  iterations~ : Int,
  epsilon~ : Float,
  limits~ : ReferenceLimits,
) -> ReadOnlyArray[Float] raise AdvancedDecoderReferenceError {
  validate_shape(width, width, limits)
  if logits.length() != width * width {
    raise InvalidDimension
  }
  validate_values(logits, limits)
  if iterations <= 0 || iterations > limits.max_iterations {
    raise InvalidIterationCount
  }
  if !finite(epsilon) || epsilon <= 0.0F {
    raise NonFiniteParameter
  }
  let mut maximum = logits[0]
  for value in logits {
    if value > maximum {
      maximum = value
    }
  }
  let matrix : Array[Float] = []
  for value in logits {
    matrix.push(@math.expf(value - maximum))
  }
  for _ in 0..<iterations {
    for row in 0..<width {
      let mut sum = 0.0F
      for column in 0..<width {
        sum += matrix[row * width + column]
      }
      if !finite(sum) || sum <= epsilon {
        raise ZeroNormalization
      }
      for column in 0..<width {
        matrix[row * width + column] /= sum
      }
    }
    for column in 0..<width {
      let mut sum = 0.0F
      for row in 0..<width {
        sum += matrix[row * width + column]
      }
      if !finite(sum) || sum <= epsilon {
        raise ZeroNormalization
      }
      for row in 0..<width {
        matrix[row * width + column] /= sum
      }
    }
  }
  ReadOnlyArray::from_array(matrix)
}

///|
/// Applies a square row-major stream-mixing matrix to flat row-major stream
/// activations. The matrix is explicit so callers may use Sinkhorn output or a
/// hand-computed fixture without hidden normalization.
pub fn mix_hyper_connections(
  streams : ArrayView[Float],
  mixing_matrix : ArrayView[Float],
  stream_count~ : Int,
  feature_width~ : Int,
  limits~ : ReferenceLimits,
) -> ReadOnlyArray[Float] raise AdvancedDecoderReferenceError {
  validate_shape(stream_count, feature_width, limits)
  if stream_count > limits.max_width ||
    streams.length() != stream_count * feature_width ||
    mixing_matrix.length() != stream_count * stream_count ||
    !combined_values_within_limit(
      streams.length(),
      mixing_matrix.length(),
      0,
      limits,
    ) {
    raise InvalidDimension
  }
  validate_values(streams, limits)
  validate_values(mixing_matrix, limits)
  let output = Array::make(stream_count * feature_width, 0.0F)
  for destination in 0..<stream_count {
    for source in 0..<stream_count {
      let weight = mixing_matrix[destination * stream_count + source]
      for column in 0..<feature_width {
        output[destination * feature_width + column] += weight *
          streams[source * feature_width + column]
      }
    }
  }
  for index, value in output {
    if !finite(value) {
      raise NonFiniteInput(index)
    }
  }
  ReadOnlyArray::from_array(output)
}
