///|
pub fn TensorParallelRankConfigureFrameBuffer::new(
  binding : @rank_group_wire.RankGroupWireBinding,
) -> TensorParallelRankConfigureFrameBuffer {
  {
    binding,
    storage: FixedArray::make(RANK_CONFIGURE_CAPACITY, b'\x00'),
    epoch: 0UL,
    length: 0,
  }
}

///|
pub fn TensorParallelRankConfigureFrameBuffer::capacity(
  self : TensorParallelRankConfigureFrameBuffer,
) -> Int {
  self.storage.length()
}

///|
fn TensorParallelRankConfigureFrameBuffer::publish(
  self : TensorParallelRankConfigureFrameBuffer,
  contract : TensorParallelRankConfigureContract,
) -> ValidatedTensorParallelRankConfigureFrame raise TensorParallelRankConfigureError {
  if self.epoch == 0xffffffffffffffffUL {
    raise Invalid(FrameEpoch)
  }
  if !configure_bindings_equal(self.binding, contract.binding) {
    raise Invalid(Binding)
  }
  contract.copy_to(self.storage, destination_offset=0)
  self.epoch += 1UL
  self.length = contract.length
  ValidatedTensorParallelRankConfigureFrame(self, self.epoch)
}

///|
pub fn TensorParallelRankConfigureFrameBuffer::encode(
  self : TensorParallelRankConfigureFrameBuffer,
  contract : TensorParallelRankConfigureContract,
) -> ValidatedTensorParallelRankConfigureFrame raise TensorParallelRankConfigureError {
  self.publish(contract)
}

///|
pub fn TensorParallelRankConfigureFrameBuffer::load(
  self : TensorParallelRankConfigureFrameBuffer,
  source : FixedArray[Byte],
  length : Int,
) -> ValidatedTensorParallelRankConfigureFrame raise TensorParallelRankConfigureError {
  let contract = decode_rank_configure(source, length, self.binding)
  self.publish(contract)
}

///|
fn ValidatedTensorParallelRankConfigureFrame::owner(
  self : ValidatedTensorParallelRankConfigureFrame,
) -> TensorParallelRankConfigureFrameBuffer raise TensorParallelRankConfigureError {
  if self.1 == 0UL || self.0.epoch != self.1 || self.0.length == 0 {
    raise StaleFrame
  }
  self.0
}

///|
pub fn ValidatedTensorParallelRankConfigureFrame::length(
  self : ValidatedTensorParallelRankConfigureFrame,
) -> Int raise TensorParallelRankConfigureError {
  self.owner().length
}

///|
pub fn ValidatedTensorParallelRankConfigureFrame::contract(
  self : ValidatedTensorParallelRankConfigureFrame,
) -> TensorParallelRankConfigureContract raise TensorParallelRankConfigureError {
  let owner = self.owner()
  decode_rank_configure(owner.storage, owner.length, owner.binding)
}

///|
pub fn ValidatedTensorParallelRankConfigureFrame::copy_to(
  self : ValidatedTensorParallelRankConfigureFrame,
  destination : FixedArray[Byte],
  destination_offset~ : Int,
) -> Unit raise TensorParallelRankConfigureError {
  let owner = self.owner()
  if destination_offset < 0 ||
    destination_offset > destination.length() - owner.length {
    raise Invalid(Destination)
  }
  for index in 0..<owner.length {
    destination[destination_offset + index] = owner.storage[index]
  }
}
