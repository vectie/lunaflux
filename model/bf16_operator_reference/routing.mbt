///|
/// Prepared scratch for the full-softmax BF16 router. Ties use lower expert
/// IDs; upstream topk tie order is unspecified. No allocation in run_into.
pub struct Router {
  priv plan : @precision.SoftmaxTopKPrecision
  priv probabilities : FixedArray[Float]
  priv top : FixedArray[Float]
}

///|
pub fn Router::new(plan : @precision.SoftmaxTopKPrecision) -> Router {
  let (experts, selected, _) = plan.geometry()
  {
    plan,
    probabilities: FixedArray::make(experts, 0.0F),
    top: FixedArray::make(selected, 0.0F),
  }
}

///|
pub fn Router::run_into(
  self : Router,
  logits : ArrayView[Float],
  indices : FixedArray[Int32],
  weights : FixedArray[Float],
  rows~ : Int32,
) -> Unit raise {
  let (experts, selected, maximum_rows) = self.plan.geometry()
  guard rows > 0 &&
    rows <= maximum_rows &&
    logits.length().to_int64() == rows.to_int64() * experts.to_int64() &&
    indices.length().to_int64() == rows.to_int64() * selected.to_int64() &&
    weights.length() == indices.length() else {
    fail("invalid routing buffers")
  }
  for row in 0..<rows {
    let base = row * experts
    let out = row * selected
    let mut maximum = Float::reinterpret_from_uint(0xff800000U)
    for expert in 0..<experts {
      let value = round_bf16(logits[base + expert])
      guard !value.is_nan() && !value.is_inf() else {
        fail("routing expects finite BF16 logits")
      }
      maximum = maximum.max(value)
    }
    let mut total = 0.0F
    for expert in 0..<experts {
      let value = exp32(round_bf16(logits[base + expert]) - maximum)
      self.probabilities[expert] = value
      total += value
    }
    for expert in 0..<experts {
      self.probabilities[expert] /= total
    }
    for rank in 0..<selected {
      let mut best = 0
      for expert in 1..<experts {
        if self.probabilities[expert] > self.probabilities[best] {
          best = expert
        }
      }
      indices[out + rank] = best
      self.top[rank] = self.probabilities[best]
      self.probabilities[best] = -1.0F
    }
    let mut selected_total = 0.0F
    for rank in 0..<selected {
      selected_total += self.top[rank]
    }
    for rank in 0..<selected {
      weights[out + rank] = round_bf16(self.top[rank] / selected_total)
    }
  }
}
