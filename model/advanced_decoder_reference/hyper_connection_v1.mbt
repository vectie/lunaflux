///|
fn hc_checked_geometry(
  row_count : Int,
  stream_count : Int,
  feature_width : Int,
  limits : ReferenceLimits,
) -> Int raise AdvancedDecoderReferenceError {
  if row_count <= 0 ||
    stream_count <= 1 ||
    feature_width <= 0 ||
    stream_count > limits.max_width ||
    feature_width > limits.max_width ||
    row_count > limits.max_values / stream_count ||
    row_count * stream_count > limits.max_values / feature_width {
    raise InvalidDimension
  }
  stream_count * feature_width
}

///|
fn hc_sigmoid(value : Float) -> Float {
  if value >= 0.0F {
    1.0F / (1.0F + @math.expf(-value))
  } else {
    let exponential = @math.expf(value)
    exponential / (1.0F + exponential)
  }
}

///|
fn hc_normalized_mixes(
  streams : ArrayView[Float],
  function_weights : ArrayView[Float],
  row_count : Int,
  stream_count : Int,
  feature_width : Int,
  control_count : Int,
  normalization_epsilon : Float,
) -> Array[Float] {
  let flattened = stream_count * feature_width
  let mixes = Array::make(row_count * control_count, 0.0F)
  for row in 0..<row_count {
    let input_base = row * flattened
    let mut square_sum = 0.0F
    for column in 0..<flattened {
      let value = streams[input_base + column]
      square_sum += value * value
    }
    let inverse = 1.0F /
      (square_sum / Float::from_int(flattened) + normalization_epsilon).sqrt()
    for control in 0..<control_count {
      let weight_base = control * flattened
      let mut dot = 0.0F
      for column in 0..<flattened {
        dot += streams[input_base + column] *
          function_weights[weight_base + column]
      }
      mixes[row * control_count + control] = dot * inverse
    }
  }
  mixes
}

///|
/// Exact official block control V1 semantics. `streams` are values widened
/// from BF16; function/base/scale and all returned controls are F32. Function
/// weights are `[6*stream_count, stream_count*feature_width]`, base is
/// `[6*stream_count]`, and scale is `[3]`.
pub fn hyper_connection_block_controls_v1(
  streams : ArrayView[Float],
  function_weights : ArrayView[Float],
  base : ArrayView[Float],
  scale : ArrayView[Float],
  row_count~ : Int,
  stream_count~ : Int,
  feature_width~ : Int,
  normalization_epsilon~ : Float,
  control_epsilon~ : Float,
  sinkhorn_iterations~ : Int,
  limits~ : ReferenceLimits,
) -> HyperConnectionBlockControls raise AdvancedDecoderReferenceError {
  let flattened = hc_checked_geometry(
    row_count, stream_count, feature_width, limits,
  )
  let control_count = 6 * stream_count
  if streams.length() != row_count * flattened ||
    control_count > limits.max_width ||
    function_weights.length() != control_count * flattened ||
    base.length() != control_count ||
    scale.length() != 3 {
    raise InvalidDimension
  }
  if sinkhorn_iterations <= 0 || sinkhorn_iterations > limits.max_iterations {
    raise InvalidIterationCount
  }
  if !finite(normalization_epsilon) ||
    normalization_epsilon <= 0.0F ||
    !finite(control_epsilon) ||
    control_epsilon <= 0.0F {
    raise NonFiniteParameter
  }
  validate_values(streams, limits)
  validate_values(function_weights, limits)
  validate_values(base, limits)
  validate_values(scale, limits)
  let mixes = hc_normalized_mixes(
    streams, function_weights, row_count, stream_count, feature_width, control_count,
    normalization_epsilon,
  )
  let pre = Array::make(row_count * stream_count, 0.0F)
  let post = Array::make(row_count * stream_count, 0.0F)
  let combination = Array::make(row_count * stream_count * stream_count, 0.0F)
  for row in 0..<row_count {
    let mix_base = row * control_count
    for stream in 0..<stream_count {
      pre[row * stream_count + stream] = hc_sigmoid(
          mixes[mix_base + stream] * scale[0] + base[stream],
        ) +
        control_epsilon
      post[row * stream_count + stream] = 2.0F *
        hc_sigmoid(
          mixes[mix_base + stream_count + stream] * scale[1] +
          base[stream_count + stream],
        )
    }
    let matrix_base = row * stream_count * stream_count
    for destination in 0..<stream_count {
      let mut maximum = mixes[mix_base +
        2 * stream_count +
        destination * stream_count] *
        scale[2] +
        base[2 * stream_count + destination * stream_count]
      for source in 1..<stream_count {
        let index = 2 * stream_count + destination * stream_count + source
        let value = mixes[mix_base + index] * scale[2] + base[index]
        if value > maximum {
          maximum = value
        }
      }
      let mut sum = 0.0F
      for source in 0..<stream_count {
        let index = 2 * stream_count + destination * stream_count + source
        let value = @math.expf(
          mixes[mix_base + index] * scale[2] + base[index] - maximum,
        )
        combination[matrix_base + destination * stream_count + source] = value
        sum += value
      }
      for source in 0..<stream_count {
        let index = matrix_base + destination * stream_count + source
        combination[index] = combination[index] / sum + control_epsilon
      }
    }
    for iteration in 0..<sinkhorn_iterations {
      if iteration > 0 {
        for destination in 0..<stream_count {
          let mut sum = 0.0F
          for source in 0..<stream_count {
            sum += combination[matrix_base + destination * stream_count + source]
          }
          for source in 0..<stream_count {
            let index = matrix_base + destination * stream_count + source
            combination[index] /= sum + control_epsilon
          }
        }
      }
      for source in 0..<stream_count {
        let mut sum = 0.0F
        for destination in 0..<stream_count {
          sum += combination[matrix_base + destination * stream_count + source]
        }
        for destination in 0..<stream_count {
          let index = matrix_base + destination * stream_count + source
          combination[index] /= sum + control_epsilon
        }
      }
    }
  }
  {
    pre: ReadOnlyArray::from_array(pre),
    post: ReadOnlyArray::from_array(post),
    combination: ReadOnlyArray::from_array(combination),
  }
}

///|
/// Official pre-reduction V1. Inputs are BF16-boundary stream values widened
/// to Float plus F32 controls; returned values are to be rounded once to BF16.
pub fn hyper_connection_pre_reduce_v1(
  streams : ArrayView[Float],
  pre : ArrayView[Float],
  row_count~ : Int,
  stream_count~ : Int,
  feature_width~ : Int,
  limits~ : ReferenceLimits,
) -> ReadOnlyArray[Float] raise AdvancedDecoderReferenceError {
  let flattened = hc_checked_geometry(
    row_count, stream_count, feature_width, limits,
  )
  if streams.length() != row_count * flattened ||
    pre.length() != row_count * stream_count {
    raise InvalidDimension
  }
  validate_values(streams, limits)
  validate_values(pre, limits)
  let output = Array::make(row_count * feature_width, 0.0F)
  for row in 0..<row_count {
    for column in 0..<feature_width {
      let mut sum = 0.0F
      for stream in 0..<stream_count {
        sum += pre[row * stream_count + stream] *
          streams[(row * stream_count + stream) * feature_width + column]
      }
      output[row * feature_width + column] = sum
    }
  }
  ReadOnlyArray::from_array(output)
}

///|
/// Official post-residual combination V1. Both activation inputs are BF16
/// boundary values widened to Float; controls are F32 and output rounds to BF16.
pub fn hyper_connection_post_combine_v1(
  branch : ArrayView[Float],
  residual : ArrayView[Float],
  post : ArrayView[Float],
  combination : ArrayView[Float],
  row_count~ : Int,
  stream_count~ : Int,
  feature_width~ : Int,
  limits~ : ReferenceLimits,
) -> ReadOnlyArray[Float] raise AdvancedDecoderReferenceError {
  let flattened = hc_checked_geometry(
    row_count, stream_count, feature_width, limits,
  )
  if branch.length() != row_count * feature_width ||
    residual.length() != row_count * flattened ||
    post.length() != row_count * stream_count ||
    combination.length() != row_count * stream_count * stream_count {
    raise InvalidDimension
  }
  validate_values(branch, limits)
  validate_values(residual, limits)
  validate_values(post, limits)
  validate_values(combination, limits)
  let output = Array::make(row_count * flattened, 0.0F)
  for row in 0..<row_count {
    for destination in 0..<stream_count {
      for column in 0..<feature_width {
        let mut value = post[row * stream_count + destination] *
          branch[row * feature_width + column]
        for source in 0..<stream_count {
          value += combination[(row * stream_count + destination) * stream_count +
            source] *
            residual[(row * stream_count + source) * feature_width + column]
        }
        output[(row * stream_count + destination) * feature_width + column] = value
      }
    }
  }
  ReadOnlyArray::from_array(output)
}

///|
/// Official head reduction V1, including F32 normalized control projection and
/// sigmoid weights. Inputs are BF16-boundary values widened to Float; output
/// is to be rounded once to BF16.
pub fn hyper_connection_head_reduce_v1(
  streams : ArrayView[Float],
  function_weights : ArrayView[Float],
  base : ArrayView[Float],
  scale : Float,
  row_count~ : Int,
  stream_count~ : Int,
  feature_width~ : Int,
  normalization_epsilon~ : Float,
  control_epsilon~ : Float,
  limits~ : ReferenceLimits,
) -> ReadOnlyArray[Float] raise AdvancedDecoderReferenceError {
  let flattened = hc_checked_geometry(
    row_count, stream_count, feature_width, limits,
  )
  if streams.length() != row_count * flattened ||
    function_weights.length() != stream_count * flattened ||
    base.length() != stream_count {
    raise InvalidDimension
  }
  if !finite(scale) ||
    !finite(normalization_epsilon) ||
    normalization_epsilon <= 0.0F ||
    !finite(control_epsilon) ||
    control_epsilon <= 0.0F {
    raise NonFiniteParameter
  }
  validate_values(streams, limits)
  validate_values(function_weights, limits)
  validate_values(base, limits)
  let mixes = hc_normalized_mixes(
    streams, function_weights, row_count, stream_count, feature_width, stream_count,
    normalization_epsilon,
  )
  let pre = Array::make(row_count * stream_count, 0.0F)
  for row in 0..<row_count {
    for stream in 0..<stream_count {
      pre[row * stream_count + stream] = hc_sigmoid(
          mixes[row * stream_count + stream] * scale + base[stream],
        ) +
        control_epsilon
    }
  }
  hyper_connection_pre_reduce_v1(
    streams,
    pre,
    row_count~,
    stream_count~,
    feature_width~,
    limits~,
  )
}
