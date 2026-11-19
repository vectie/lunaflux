///|
pub suberror DeltaError {
  InvalidDelta(String)
} derive(Debug)

///|
pub extend DeltaError with @debug.Debug::{to_repr}

///|
/// Prepared scalar FP32 reference for Qwen3.5-family gated delta attention.
/// Model-owned BF16 input/output boundaries remain outside this operator.
/// This is an offline reference, never a replacement for a missing CUDA kernel.
pub struct State {
  priv geometry : @hybrid.RecurrentAttention
  priv matrix : FixedArray[Float]
  priv query : FixedArray[Float]
  priv key : FixedArray[Float]
  priv delta : FixedArray[Float]
}

///|
pub fn State::new(
  geometry : @hybrid.RecurrentAttention,
) -> State raise DeltaError {
  guard [
      geometry.key_heads,
      geometry.key_width,
      geometry.value_heads,
      geometry.value_width,
      geometry.convolution_width,
    ].all(x => x > 0) &&
    geometry.value_heads % geometry.key_heads == 0 else {
    raise InvalidDelta("invalid recurrent dimensions")
  }
  let elements = geometry.value_heads.to_int64() * geometry.key_width.to_int64()
  guard elements <= 2147483647L / geometry.value_width.to_int64() &&
    geometry.key_heads.to_int64() * geometry.key_width.to_int64() <= 2147483647L else {
    raise InvalidDelta("recurrent capacity overflow")
  }
  {
    geometry,
    matrix: FixedArray::make(
      (elements * geometry.value_width.to_int64()).to_int(),
      0.0F,
    ),
    query: FixedArray::make(geometry.key_heads * geometry.key_width, 0.0F),
    key: FixedArray::make(geometry.key_heads * geometry.key_width, 0.0F),
    delta: FixedArray::make(geometry.value_width, 0.0F),
  }
}

///|
pub fn State::state_elements(self : State) -> Int32 {
  self.matrix.length()
}

///|
pub fn State::reset(self : State) -> Unit {
  self.matrix.fill(0.0F)
}

///|
fn finite(values : ArrayView[Float]) -> Bool {
  values.all(x => !x.is_nan() && !x.is_inf())
}

///|
pub fn State::restore(
  self : State,
  source : ArrayView[Float],
) -> Unit raise DeltaError {
  guard source.length() == self.matrix.length() && finite(source) else {
    raise InvalidDelta("invalid recurrent state")
  }
  for i, value in source {
    self.matrix[i] = value
  }
}

///|
pub fn State::copy_into(
  self : State,
  destination : FixedArray[Float],
) -> Unit raise DeltaError {
  guard destination.length() == self.matrix.length() else {
    raise InvalidDelta("invalid recurrent destination")
  }
  for i in 0..<self.matrix.length() {
    destination[i] = self.matrix[i]
  }
}

///|
/// One token, q/k [key_head,key_width], v/output [value_head,value_width],
/// log_decay/beta [value_head]. Q/K heads repeat consecutively across value
/// heads. State is [value_head,key_width,value_width], with FP32 updates.
/// Q/K use rsqrt(sum(x*x)+1e-6); query also scales by 1/sqrt(key_width).
/// Decay happens before the memory read and rank-one delta update. The gate
/// is scalar per value head, unlike Kimi Delta's per-key-channel decay.
pub fn State::run_into(
  self : State,
  query : ArrayView[Float],
  key : ArrayView[Float],
  value : ArrayView[Float],
  log_decay : ArrayView[Float],
  beta : ArrayView[Float],
  output : FixedArray[Float],
) -> Unit raise DeltaError {
  let g = self.geometry
  guard query.length() == self.query.length() &&
    key.length() == self.key.length() &&
    value.length() == g.value_heads * g.value_width &&
    output.length() == value.length() &&
    log_decay.length() == g.value_heads &&
    beta.length() == g.value_heads &&
    finite(query) &&
    finite(key) &&
    finite(value) &&
    finite(beta) &&
    log_decay.all(x => !x.is_nan() && x <= 0.0F) &&
    beta.all(x => x >= 0.0F && x <= 1.0F) else {
    raise InvalidDelta("invalid recurrent inputs")
  }
  let query_scale = 1.0F / Float::from_int(g.key_width).sqrt()
  for head in 0..<g.key_heads {
    let base = head * g.key_width
    let mut q_squared = 0.0F
    let mut k_squared = 0.0F
    for channel in 0..<g.key_width {
      q_squared += query[base + channel] * query[base + channel]
      k_squared += key[base + channel] * key[base + channel]
    }
    let q_scale = 1.0F / (q_squared + 0.000001F).sqrt()
    let k_scale = 1.0F / (k_squared + 0.000001F).sqrt()
    for channel in 0..<g.key_width {
      self.query[base + channel] = query[base + channel] * q_scale * query_scale
      self.key[base + channel] = key[base + channel] * k_scale
    }
  }
  let repeat = g.value_heads / g.key_heads
  for head in 0..<g.value_heads {
    let key_base = head / repeat * g.key_width
    let value_base = head * g.value_width
    let state_base = head * g.key_width * g.value_width
    let decay = Float::from_double(@math.exp(log_decay[head].to_double()))
    for i in 0..<(g.key_width * g.value_width) {
      self.matrix[state_base + i] *= decay
    }
    for channel in 0..<g.value_width {
      let mut memory = 0.0F
      for k in 0..<g.key_width {
        memory += self.matrix[state_base + k * g.value_width + channel] *
          self.key[key_base + k]
      }
      self.delta[channel] = (value[value_base + channel] - memory) * beta[head]
    }
    for k in 0..<g.key_width {
      for channel in 0..<g.value_width {
        let index = state_base + k * g.value_width + channel
        self.matrix[index] += self.key[key_base + k] * self.delta[channel]
      }
    }
    for channel in 0..<g.value_width {
      let mut result = 0.0F
      for k in 0..<g.key_width {
        result += self.matrix[state_base + k * g.value_width + channel] *
          self.query[key_base + k]
      }
      output[value_base + channel] = result
    }
  }
}
