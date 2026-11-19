///|
/// Channel-local short convolution history, oldest first. Only K-1 samples are
/// needed by this scalar reference; the native backend may pad its cache to K.
pub struct Convolution {
  priv channels : Int32
  priv kernel : Int32
  priv history : FixedArray[Float]
  priv projection_boundary : @precision.ConvolutionProjectionBoundary
}

///|
pub fn Convolution::new(
  channels : Int32,
  kernel : Int32,
  projection_boundary~ : @precision.ConvolutionProjectionBoundary = Float32BeforeSilu,
) -> Convolution raise DeltaError {
  guard channels > 0 &&
    kernel > 0 &&
    channels.to_int64() * kernel.to_int64() <= 2147483647L else {
    raise InvalidDelta("invalid convolution geometry")
  }
  {
    channels,
    kernel,
    history: FixedArray::make(channels * (kernel - 1), 0.0F),
    projection_boundary,
  }
}

///|
pub fn Convolution::reset(self : Convolution) -> Unit {
  self.history.fill(0.0F)
}

///|
pub fn Convolution::copy_into(
  self : Convolution,
  destination : FixedArray[Float],
) -> Unit raise DeltaError {
  guard destination.length() == self.history.length() else {
    raise InvalidDelta("invalid convolution destination")
  }
  for i in 0..<self.history.length() {
    destination[i] = self.history[i]
  }
}

///|
pub fn Convolution::restore(
  self : Convolution,
  source : ArrayView[Float],
) -> Unit raise DeltaError {
  guard source.length() == self.history.length() && finite(source) else {
    raise InvalidDelta("invalid convolution history")
  }
  for i, value in source {
    self.history[i] = value
  }
}

///|
/// Weights [channel,kernel], one input/output value per channel, SiLU after
/// the FP32 dot and optional BF16 projection round. Caller applies final BF16
/// output rounding and supplies BF16-valued inputs/weights for that contract.
pub fn Convolution::run_into(
  self : Convolution,
  input : ArrayView[Float],
  weights : ArrayView[Float],
  output : FixedArray[Float],
) -> Unit raise DeltaError {
  guard input.length() == self.channels &&
    weights.length() == self.channels * self.kernel &&
    output.length() == self.channels else {
    raise InvalidDelta("invalid convolution buffers")
  }
  let history = self.kernel - 1
  for channel in 0..<self.channels {
    let base = channel * history
    let weight_base = channel * self.kernel
    let current = input[channel]
    let mut sum = 0.0F
    for i in 0..<history {
      sum += self.history[base + i] * weights[weight_base + i]
    }
    sum += current * weights[weight_base + history]
    if self.projection_boundary is Bf16BeforeSilu {
      sum = @bf16.round_bf16(sum)
    }
    output[channel] = sum /
      (1.0F + Float::from_double(@math.exp((-sum).to_double())))
    if history > 0 {
      for i in 0..<(history - 1) {
        self.history[base + i] = self.history[base + i + 1]
      }
      self.history[base + history - 1] = current
    }
  }
}
