///|
pub fn DeepSeekPredictionAttention::persistent_state_bytes(
  self : Self,
) -> Int64 {
  self.ring.persistent_state_bytes()
}

///|
pub fn DeepSeekPredictionAttention::persistent_state(
  self : Self,
) -> Array[@device.Allocation] raise {
  guard self.state is Prepared else {
    raise @device.DeviceError::InvalidArgument(DeviceCopyCreate)
  }
  self.ring.persistent_state()
}

///|
pub fn DeepSeekPredictionBlock::persistent_state_bytes(self : Self) -> Int64 {
  self.attention.persistent_state_bytes()
}

///|
pub fn DeepSeekPredictionBlock::persistent_state(
  self : Self,
) -> Array[@device.Allocation] raise {
  guard self.state is Prepared else {
    raise @device.DeviceError::InvalidArgument(DeviceCopyCreate)
  }
  self.attention.persistent_state()
}

///|
pub fn DeepSeekPredictor::persistent_state_bytes(self : Self) -> Int64 {
  self.blocks.fold(init=0L, (total, block) => {
    total + block.persistent_state_bytes()
  })
}

///|
/// Committed-main rings for every prediction stage. Draft residuals, logits,
/// confidence and launch descriptors are scratch; RNG is owned by the caller.
pub fn DeepSeekPredictor::persistent_state(
  self : Self,
) -> Array[@device.Allocation] raise {
  guard self.state is Prepared else {
    raise @device.DeviceError::InvalidArgument(DeviceCopyCreate)
  }
  let buffers = []
  for block in self.blocks {
    buffers.append(block.persistent_state())
  }
  buffers
}
