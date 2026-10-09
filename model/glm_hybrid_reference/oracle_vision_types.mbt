///|
pub struct GlmTinyVisionBlockPlan {
  priv patch_count : Int
  priv head_count : Int
  priv head_dimension : Int
  priv intermediate_size : Int
  priv merge_size : Int
  priv project_width : Int
  priv rms_epsilon : Double
}

///|
pub fn GlmTinyVisionBlockPlan::new(
  patch_count~ : Int,
  head_count~ : Int,
  head_dimension~ : Int,
  intermediate_size~ : Int,
  merge_size~ : Int,
  project_width~ : Int,
  rms_epsilon~ : Double,
  limits~ : GlmTinyOracleLimits,
) -> GlmTinyVisionBlockPlan raise GlmHybridReferenceError {
  if head_count <= 0 ||
    head_dimension <= 0 ||
    head_count > limits.maximum_width / head_dimension {
    raise InvalidVisionGeometry
  }
  let width = head_count * head_dimension
  if patch_count <= 0 ||
    patch_count > limits.maximum_tokens ||
    width > limits.maximum_width ||
    intermediate_size <= 0 ||
    intermediate_size > limits.maximum_intermediate ||
    merge_size <= 0 ||
    merge_size > patch_count ||
    patch_count % (merge_size * merge_size) != 0 ||
    project_width <= 0 ||
    project_width > limits.maximum_width ||
    !finite(rms_epsilon) ||
    rms_epsilon <= 0.0 {
    raise InvalidVisionGeometry
  }
  if patch_count > limits.maximum_values / width {
    raise InvalidVisionGeometry
  }
  {
    patch_count,
    head_count,
    head_dimension,
    intermediate_size,
    merge_size,
    project_width,
    rms_epsilon,
  }
}

///|
pub struct GlmTinyVisionBlockWeights {
  priv attention_norm : ReadOnlyArray[Double]
  priv query : ReadOnlyArray[Double]
  priv key : ReadOnlyArray[Double]
  priv value : ReadOnlyArray[Double]
  priv attention_output : ReadOnlyArray[Double]
  priv ffn_norm : ReadOnlyArray[Double]
  priv gate : ReadOnlyArray[Double]
  priv up : ReadOnlyArray[Double]
  priv down : ReadOnlyArray[Double]
  priv merge : ReadOnlyArray[Double]
  priv merge_bias : ReadOnlyArray[Double]
  priv project : ReadOnlyArray[Double]
  priv project_bias : ReadOnlyArray[Double]
}

///|
pub fn GlmTinyVisionBlockWeights::new(
  attention_norm~ : ArrayView[Double],
  query~ : ArrayView[Double],
  key~ : ArrayView[Double],
  value~ : ArrayView[Double],
  attention_output~ : ArrayView[Double],
  ffn_norm~ : ArrayView[Double],
  gate~ : ArrayView[Double],
  up~ : ArrayView[Double],
  down~ : ArrayView[Double],
  merge~ : ArrayView[Double],
  merge_bias~ : ArrayView[Double],
  project~ : ArrayView[Double],
  project_bias~ : ArrayView[Double],
) -> GlmTinyVisionBlockWeights {
  {
    attention_norm: ReadOnlyArray::from_array(attention_norm),
    query: ReadOnlyArray::from_array(query),
    key: ReadOnlyArray::from_array(key),
    value: ReadOnlyArray::from_array(value),
    attention_output: ReadOnlyArray::from_array(attention_output),
    ffn_norm: ReadOnlyArray::from_array(ffn_norm),
    gate: ReadOnlyArray::from_array(gate),
    up: ReadOnlyArray::from_array(up),
    down: ReadOnlyArray::from_array(down),
    merge: ReadOnlyArray::from_array(merge),
    merge_bias: ReadOnlyArray::from_array(merge_bias),
    project: ReadOnlyArray::from_array(project),
    project_bias: ReadOnlyArray::from_array(project_bias),
  }
}
