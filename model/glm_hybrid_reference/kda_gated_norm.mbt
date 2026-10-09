///|
pub fn kda_sigmoid_gated_rms_norm(
  recurrent : ArrayView[Double],
  gate : ArrayView[Double],
  weight : ArrayView[Double],
  row_count~ : Int,
  head_count~ : Int,
  head_dimension~ : Int,
  epsilon~ : Double,
  limits~ : GlmHybridReferenceLimits,
) -> GlmKdaGatedNormResult raise GlmHybridReferenceError {
  limits.check_sequence(row_count)
  limits.check_heads(head_count, head_dimension)
  if epsilon.is_nan() || epsilon.is_inf() || epsilon <= 0.0 || epsilon > 1.0 {
    raise InvalidProbability
  }
  let cells = row_count * head_count * head_dimension
  if recurrent.length() != cells ||
    gate.length() != cells ||
    weight.length() != head_dimension {
    raise InvalidShape
  }
  require_finite(recurrent)
  require_finite(gate)
  require_finite(weight)
  let values = Array::make(cells, 0.0)
  for row in 0..<row_count {
    for head in 0..<head_count {
      let base = (row * head_count + head) * head_dimension
      let mut sum_sq = 0.0
      for column in 0..<head_dimension {
        let value = recurrent[base + column]
        sum_sq = sum_sq + value * value
      }
      let inverse = 1.0 / (sum_sq / head_dimension.to_double() + epsilon).sqrt()
      for column in 0..<head_dimension {
        let normalized = recurrent[base + column] * inverse
        let weighted = weight[column] * normalized
        let sigmoid_gate = 1.0 / (1.0 + @math.exp(-gate[base + column]))
        values[base + column] = weighted * sigmoid_gate
      }
    }
  }
  {
    row_count,
    head_count,
    head_dimension,
    values: ReadOnlyArray::from_array(values),
  }
}

///|
pub fn GlmKdaGatedNormResult::row_count(self : Self) -> Int {
  self.row_count
}

///|
pub fn GlmKdaGatedNormResult::head_count(self : Self) -> Int {
  self.head_count
}

///|
pub fn GlmKdaGatedNormResult::head_dimension(self : Self) -> Int {
  self.head_dimension
}

///|
pub fn GlmKdaGatedNormResult::values(self : Self) -> ReadOnlyArray[Double] {
  self.values
}
