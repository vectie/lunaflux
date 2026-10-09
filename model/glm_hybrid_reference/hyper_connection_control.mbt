///|
fn hyper_sigmoid(value : Double) -> Double {
  1.0 / (1.0 + @math.exp(-value))
}

///|
/// Tiny scalar oracle for GLM's staged mHC function boundary. It preserves the
/// destination/source matrix order and leaves Sinkhorn normalization separate.
pub fn hyper_connection_function_and_collapse(
  streams : ArrayView[Double],
  function_weights : ArrayView[Double],
  base : ArrayView[Double],
  scale : ArrayView[Double],
  row_count~ : Int,
  stream_count~ : Int,
  hidden_width~ : Int,
  control_epsilon~ : Double,
  normalization_epsilon~ : Double,
  limits~ : GlmHybridReferenceLimits,
) -> GlmHyperConnectionFunctionResult raise GlmHybridReferenceError {
  limits.check_sequence(row_count)
  limits.check_heads(stream_count, 1)
  limits.check_channels(hidden_width)
  let flattened = stream_count * hidden_width
  let mix_width = (2 + stream_count) * stream_count
  if !finite(control_epsilon) ||
    control_epsilon <= 0.0 ||
    !finite(normalization_epsilon) ||
    normalization_epsilon <= 0.0 ||
    streams.length() != row_count * flattened ||
    function_weights.length() != mix_width * flattened ||
    base.length() != mix_width ||
    scale.length() != 3 {
    raise InvalidShape
  }
  require_finite(streams)
  require_finite(function_weights)
  require_finite(base)
  require_finite(scale)
  let post = Array::make(row_count * stream_count, 0.0)
  let initial = Array::make(row_count * stream_count * stream_count, 0.0)
  let collapsed = Array::make(row_count * hidden_width, 0.0)
  for row in 0..<row_count {
    let input_base = row * flattened
    let mut square_sum = 0.0
    for column in 0..<flattened {
      let value = streams[input_base + column]
      square_sum += value * value
    }
    let inverse = 1.0 /
      (square_sum / flattened.to_double() + normalization_epsilon).sqrt()
    let mixes = Array::make(mix_width, 0.0)
    for output in 0..<mix_width {
      let weight_base = output * flattened
      let mut dot = 0.0
      for column in 0..<flattened {
        dot += streams[input_base + column] *
          inverse *
          function_weights[weight_base + column]
      }
      mixes[output] = dot
    }
    let pre = Array::make(stream_count, 0.0)
    let vector_base = row * stream_count
    for stream in 0..<stream_count {
      pre[stream] = hyper_sigmoid(mixes[stream] * scale[0] + base[stream]) +
        control_epsilon
      let post_index = stream_count + stream
      post[vector_base + stream] = 2.0 *
        hyper_sigmoid(mixes[post_index] * scale[1] + base[post_index])
    }
    let matrix_base = row * stream_count * stream_count
    for destination in 0..<stream_count {
      let control_base = 2 * stream_count + destination * stream_count
      let mut maximum = mixes[control_base] * scale[2] + base[control_base]
      for source in 1..<stream_count {
        let index = control_base + source
        let value = mixes[index] * scale[2] + base[index]
        if value > maximum {
          maximum = value
        }
      }
      let mut sum = 0.0
      for source in 0..<stream_count {
        let control_index = control_base + source
        let value = @math.exp(
          mixes[control_index] * scale[2] + base[control_index] - maximum,
        )
        initial[matrix_base + destination * stream_count + source] = value
        sum += value
      }
      for source in 0..<stream_count {
        let index = matrix_base + destination * stream_count + source
        initial[index] = initial[index] / sum + control_epsilon
      }
    }
    for column in 0..<hidden_width {
      let mut sum = 0.0
      for stream in 0..<stream_count {
        sum += pre[stream] *
          streams[input_base + stream * hidden_width + column]
      }
      collapsed[row * hidden_width + column] = sum
    }
  }
  {
    row_count,
    stream_count,
    hidden_width,
    post: ReadOnlyArray::from_array(post),
    initial_combination: ReadOnlyArray::from_array(initial),
    collapsed: ReadOnlyArray::from_array(collapsed),
  }
}

///|
/// Tiny scalar oracle for the exact positive-matrix Sinkhorn suffix: one
/// column pass followed by `iterations - 1` ordered row/column passes.
pub fn hyper_connection_positive_sinkhorn(
  initial_combination : ArrayView[Double],
  row_count~ : Int,
  stream_count~ : Int,
  iterations~ : Int,
  epsilon~ : Double,
  limits~ : GlmHybridReferenceLimits,
) -> GlmHyperConnectionSinkhornResult raise GlmHybridReferenceError {
  limits.check_sequence(row_count)
  limits.check_heads(stream_count, 1)
  if iterations <= 0 ||
    iterations > 64 ||
    !finite(epsilon) ||
    epsilon <= 0.0 ||
    initial_combination.length() != row_count * stream_count * stream_count {
    raise InvalidShape
  }
  require_finite(initial_combination)
  let combination : Array[Double] = []
  for value in initial_combination {
    combination.push(value)
  }
  for value in combination {
    if value <= 0.0 {
      raise InvalidProbability
    }
  }
  for row in 0..<row_count {
    let matrix_base = row * stream_count * stream_count
    for source in 0..<stream_count {
      let mut sum = 0.0
      for destination in 0..<stream_count {
        sum += combination[matrix_base + destination * stream_count + source]
      }
      for destination in 0..<stream_count {
        let index = matrix_base + destination * stream_count + source
        combination[index] /= sum + epsilon
      }
    }
    for _ in 1..<iterations {
      for destination in 0..<stream_count {
        let mut sum = 0.0
        for source in 0..<stream_count {
          sum += combination[matrix_base + destination * stream_count + source]
        }
        for source in 0..<stream_count {
          let index = matrix_base + destination * stream_count + source
          combination[index] /= sum + epsilon
        }
      }
      for source in 0..<stream_count {
        let mut sum = 0.0
        for destination in 0..<stream_count {
          sum += combination[matrix_base + destination * stream_count + source]
        }
        for destination in 0..<stream_count {
          let index = matrix_base + destination * stream_count + source
          combination[index] /= sum + epsilon
        }
      }
    }
  }
  {
    row_count,
    stream_count,
    combination: ReadOnlyArray::from_array(combination),
  }
}

///|
pub fn GlmHyperConnectionFunctionResult::post(
  self : Self,
) -> ReadOnlyArray[Double] {
  self.post
}

///|
pub fn GlmHyperConnectionFunctionResult::row_count(self : Self) -> Int {
  self.row_count
}

///|
pub fn GlmHyperConnectionFunctionResult::stream_count(self : Self) -> Int {
  self.stream_count
}

///|
pub fn GlmHyperConnectionFunctionResult::hidden_width(self : Self) -> Int {
  self.hidden_width
}

///|
pub fn GlmHyperConnectionFunctionResult::initial_combination(
  self : Self,
) -> ReadOnlyArray[Double] {
  self.initial_combination
}

///|
pub fn GlmHyperConnectionFunctionResult::collapsed(
  self : Self,
) -> ReadOnlyArray[Double] {
  self.collapsed
}

///|
pub fn GlmHyperConnectionSinkhornResult::combination(
  self : Self,
) -> ReadOnlyArray[Double] {
  self.combination
}

///|
pub fn GlmHyperConnectionSinkhornResult::row_count(self : Self) -> Int {
  self.row_count
}

///|
pub fn GlmHyperConnectionSinkhornResult::stream_count(self : Self) -> Int {
  self.stream_count
}
