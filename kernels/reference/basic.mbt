///|
/// Looks up one embedding row for each token ID.
pub fn embedding(
  token_ids : ArrayView[Int],
  table : Tensor,
  limits~ : ReferenceLimits,
) -> Activation raise ReferenceKernelError {
  if token_ids.is_empty() || token_ids.length() > limits.max_rows() {
    raise InvalidShape(EmbeddingTokenCount)
  }
  let output_elements = checked_element_count(
    token_ids.length(),
    table.columns,
    limits,
  )
  let output = Array::make(output_elements, 0.0F)
  for row, token_id in token_ids {
    if token_id < 0 || token_id >= table.rows {
      raise InvalidShape(EmbeddingTokenId)
    }
    let source_offset = token_id * table.columns
    let output_offset = row * table.columns
    for column in 0..<table.columns {
      output[output_offset + column] = table.values[source_offset + column]
    }
  }
  activation_from_owned(token_ids.length(), table.columns, output, limits)
}

///|
/// Applies row-wise RMS normalization using Float accumulation.
pub fn rms_norm(
  input : Activation,
  weight : Tensor,
  epsilon~ : Float,
  limits~ : ReferenceLimits,
) -> Activation raise ReferenceKernelError {
  if weight.rows != 1 || weight.columns != input.columns {
    raise InvalidShape(NormalizationWeight)
  }
  if epsilon.is_nan() || epsilon.is_inf() {
    raise NonFinite(Epsilon)
  }
  if epsilon <= 0.0F {
    raise InvalidScalar(Epsilon)
  }
  let elements = checked_element_count(input.rows, input.columns, limits)
  let output = Array::make(elements, 0.0F)
  let width = Float::from_int(input.columns)
  for row in 0..<input.rows {
    let row_offset = row * input.columns
    let mut square_sum = 0.0F
    for column in 0..<input.columns {
      let value = input.values[row_offset + column]
      square_sum += value * value
    }
    if square_sum.is_nan() || square_sum.is_inf() {
      raise NonFinite(ComputedValue)
    }
    let inverse_rms = 1.0F / (square_sum / width + epsilon).sqrt()
    for column in 0..<input.columns {
      output[row_offset + column] = input.values[row_offset + column] *
        inverse_rms *
        weight.values[column]
    }
  }
  activation_from_owned(input.rows, input.columns, output, limits)
}

///|
fn project_values(
  input : Activation,
  weight : Tensor,
  limits : ReferenceLimits,
) -> Array[Float] raise ReferenceKernelError {
  if weight.columns != input.columns {
    raise InvalidShape(ProjectionInputWidth)
  }
  let elements = checked_element_count(input.rows, weight.rows, limits)
  let output = Array::make(elements, 0.0F)
  for row in 0..<input.rows {
    let input_offset = row * input.columns
    let output_offset = row * weight.rows
    for output_column in 0..<weight.rows {
      let weight_offset = output_column * weight.columns
      let mut accumulator = 0.0F
      for input_column in 0..<input.columns {
        accumulator += input.values[input_offset + input_column] *
          weight.values[weight_offset + input_column]
      }
      if accumulator.is_nan() || accumulator.is_inf() {
        raise NonFinite(ComputedValue)
      }
      output[output_offset + output_column] = accumulator
    }
  }
  output
}

///|
/// Computes `input * transpose(weight)` with Float accumulation.
pub fn dense_projection(
  input : Activation,
  weight : Tensor,
  limits~ : ReferenceLimits,
) -> Activation raise ReferenceKernelError {
  let output = project_values(input, weight, limits)
  activation_from_owned(input.rows, weight.rows, output, limits)
}

///|
/// Computes and concatenates dense Q, K, and V projections in that order.
pub fn qkv_projection(
  input : Activation,
  query_weight : Tensor,
  key_weight : Tensor,
  value_weight : Tensor,
  limits~ : ReferenceLimits,
) -> Activation raise ReferenceKernelError {
  if query_weight.columns != input.columns ||
    key_weight.columns != input.columns ||
    value_weight.columns != input.columns {
    raise InvalidShape(QkvInputWidth)
  }
  if key_weight.rows != value_weight.rows {
    raise InvalidShape(QkvOutputWidth)
  }
  let qk_width = checked_sum(query_weight.rows, key_weight.rows)
  let output_width = checked_sum(qk_width, value_weight.rows)
  let elements = checked_element_count(input.rows, output_width, limits)
  let query = project_values(input, query_weight, limits)
  let key = project_values(input, key_weight, limits)
  let value = project_values(input, value_weight, limits)
  let output = Array::make(elements, 0.0F)
  for row in 0..<input.rows {
    let output_offset = row * output_width
    let query_offset = row * query_weight.rows
    let key_offset = row * key_weight.rows
    let value_offset = row * value_weight.rows
    for column in 0..<query_weight.rows {
      output[output_offset + column] = query[query_offset + column]
    }
    for column in 0..<key_weight.rows {
      output[output_offset + query_weight.rows + column] = key[key_offset +
        column]
    }
    for column in 0..<value_weight.rows {
      output[output_offset + query_weight.rows + key_weight.rows + column] = value[value_offset +
        column]
    }
  }
  activation_from_owned(input.rows, output_width, output, limits)
}

///|
/// Adds two identically shaped activation matrices.
pub fn residual_add(
  left : Activation,
  right : Activation,
  limits~ : ReferenceLimits,
) -> Activation raise ReferenceKernelError {
  if left.rows != right.rows || left.columns != right.columns {
    raise InvalidShape(ResidualOperands)
  }
  let elements = checked_element_count(left.rows, left.columns, limits)
  let output = Array::make(elements, 0.0F)
  for index in 0..<elements {
    output[index] = left.values[index] + right.values[index]
  }
  activation_from_owned(left.rows, left.columns, output, limits)
}

///|
/// Computes the vocabulary projection used by a language-model head.
pub fn language_model_head(
  input : Activation,
  weight : Tensor,
  limits~ : ReferenceLimits,
) -> Activation raise ReferenceKernelError {
  dense_projection(input, weight, limits~)
}
