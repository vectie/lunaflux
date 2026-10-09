///|
/// Tiny scalar semantics for the safe-bound F32 KDA forget transform. The raw
/// logits are upstream projection outputs; recurrent state evolution is not
/// part of this operation.
pub fn kda_safe_decay_control(
  raw_forget : ArrayView[Double],
  dt_bias : ArrayView[Double],
  a_log : ArrayView[Double],
  row_count~ : Int,
  head_count~ : Int,
  head_dimension~ : Int,
  lower_bound~ : Double,
  limits~ : GlmHybridReferenceLimits,
) -> GlmKdaDecayControlResult raise GlmHybridReferenceError {
  limits.check_sequence(row_count)
  limits.check_heads(head_count, head_dimension)
  if lower_bound.is_nan() || lower_bound.is_inf() || lower_bound >= 0.0 {
    raise InvalidProbability
  }
  let row_width = head_count * head_dimension
  if raw_forget.length() != row_count * row_width ||
    dt_bias.length() != row_width ||
    a_log.length() != head_count {
    raise InvalidShape
  }
  require_finite(raw_forget)
  require_finite(dt_bias)
  require_finite(a_log)
  let values = Array::make(raw_forget.length(), 0.0)
  for row in 0..<row_count {
    let row_base = row * row_width
    for head in 0..<head_count {
      let rate = @math.exp(a_log[head])
      let head_base = row_base + head * head_dimension
      let bias_base = head * head_dimension
      for component in 0..<head_dimension {
        let biased = raw_forget[head_base + component] +
          dt_bias[bias_base + component]
        let scaled = rate * biased
        let sigmoid = 1.0 / (1.0 + @math.exp(-scaled))
        values[head_base + component] = lower_bound * sigmoid
      }
    }
  }
  {
    row_count,
    head_count,
    head_dimension,
    log_decay: ReadOnlyArray::from_array(values),
  }
}

///|
pub fn GlmKdaDecayControlResult::row_count(self : Self) -> Int {
  self.row_count
}

///|
pub fn GlmKdaDecayControlResult::head_count(self : Self) -> Int {
  self.head_count
}

///|
pub fn GlmKdaDecayControlResult::head_dimension(self : Self) -> Int {
  self.head_dimension
}

///|
pub fn GlmKdaDecayControlResult::log_decay(
  self : Self,
) -> ReadOnlyArray[Double] {
  self.log_decay
}
