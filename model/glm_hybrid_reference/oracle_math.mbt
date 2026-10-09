///|
fn oracle_check_values(
  count : Int,
  limits : GlmTinyOracleLimits,
) -> Unit raise GlmHybridReferenceError {
  if count < 0 || count > limits.maximum_values {
    raise BoundsExceeded
  }
}

///|
fn oracle_require_finite(
  values : ArrayView[Double],
  limits : GlmTinyOracleLimits,
) -> Unit raise GlmHybridReferenceError {
  oracle_check_values(values.length(), limits)
  require_finite(values)
}

///|
fn oracle_linear(
  input : ArrayView[Double],
  weights : ArrayView[Double],
  rows : Int,
  input_width : Int,
  output_width : Int,
  limits : GlmTinyOracleLimits,
) -> Array[Double] raise GlmHybridReferenceError {
  if rows <= 0 ||
    input_width <= 0 ||
    output_width <= 0 ||
    input.length() != rows * input_width ||
    weights.length() != output_width * input_width {
    raise InvalidShape
  }
  oracle_require_finite(input, limits)
  oracle_require_finite(weights, limits)
  oracle_check_values(rows * output_width, limits)
  let output = Array::make(rows * output_width, 0.0)
  for row in 0..<rows {
    for destination in 0..<output_width {
      let mut sum = 0.0
      for source in 0..<input_width {
        sum += input[row * input_width + source] *
          weights[destination * input_width + source]
      }
      if !finite(sum) {
        raise NonFiniteInput
      }
      output[row * output_width + destination] = sum
    }
  }
  output
}

///|
fn oracle_low_rank_projection(
  input : ArrayView[Double],
  first : ArrayView[Double],
  second : ArrayView[Double],
  rows : Int,
  width : Int,
  rank : Int,
  limits : GlmTinyOracleLimits,
) -> Array[Double] raise GlmHybridReferenceError {
  let reduced = oracle_linear(input, first, rows, width, rank, limits)
  oracle_linear(reduced, second, rows, rank, width, limits)
}

///|
fn oracle_rms_norm(
  input : ArrayView[Double],
  scale : ArrayView[Double],
  rows : Int,
  width : Int,
  epsilon : Double,
  limits : GlmTinyOracleLimits,
) -> Array[Double] raise GlmHybridReferenceError {
  if input.length() != rows * width ||
    scale.length() != width ||
    !finite(epsilon) ||
    epsilon <= 0.0 {
    raise InvalidShape
  }
  oracle_require_finite(input, limits)
  oracle_require_finite(scale, limits)
  let output = Array::make(input.length(), 0.0)
  for row in 0..<rows {
    let mut sum_squares = 0.0
    for column in 0..<width {
      let value = input[row * width + column]
      sum_squares += value * value
    }
    let inverse = 1.0 / (sum_squares / width.to_double() + epsilon).sqrt()
    if !finite(inverse) {
      raise NonFiniteInput
    }
    for column in 0..<width {
      let value = input[row * width + column] * inverse * scale[column]
      if !finite(value) {
        raise NonFiniteInput
      }
      output[row * width + column] = value
    }
  }
  output
}

///|
fn oracle_swiglu(
  input : ArrayView[Double],
  gate : ArrayView[Double],
  up : ArrayView[Double],
  down : ArrayView[Double],
  rows : Int,
  width : Int,
  intermediate : Int,
  limits : GlmTinyOracleLimits,
) -> Array[Double] raise GlmHybridReferenceError {
  let gate_values = oracle_linear(
    input, gate, rows, width, intermediate, limits,
  )
  let up_values = oracle_linear(input, up, rows, width, intermediate, limits)
  let activated = Array::make(rows * intermediate, 0.0)
  for index in 0..<activated.length() {
    let gate_value = gate_values[index]
    let value = gate_value / (1.0 + @math.exp(-gate_value)) * up_values[index]
    if !finite(value) {
      raise NonFiniteInput
    }
    activated[index] = value
  }
  oracle_linear(activated, down, rows, intermediate, width, limits)
}

///|
fn oracle_add(
  left : ArrayView[Double],
  right : ArrayView[Double],
  limits : GlmTinyOracleLimits,
) -> Array[Double] raise GlmHybridReferenceError {
  if left.length() != right.length() {
    raise InvalidShape
  }
  oracle_require_finite(left, limits)
  oracle_require_finite(right, limits)
  let output = Array::make(left.length(), 0.0)
  for index in 0..<left.length() {
    let value = left[index] + right[index]
    if !finite(value) {
      raise NonFiniteInput
    }
    output[index] = value
  }
  output
}
