///|
priv enum SeedInputState {
  Ready
  Uploading
  Failed
  Closed
}

///|
/// Prepared execution input for an already committed token. In particular a
/// rejected suffix must not choose its seed from stale output-row metadata.
/// The parent joins this upload before submitting its prepared kernel queue.
pub struct CommittedTokenSeed {
  priv allocation : @device.Allocation
  priv pool : @device.TransferPool
  priv payload : FixedArray[Byte]
  priv vocabulary : Int
  priv mut state : SeedInputState
}

///|
pub fn CommittedTokenSeed::new(
  context : @device.Context,
  vocabulary_size~ : Int,
) -> Self raise {
  guard vocabulary_size > 0 else {
    raise @device.DeviceError::InvalidArgument(StreamingTransfer)
  }
  let allocation = context.allocate(4L)
  errdefer allocation.close()
  let pool = context.prepare_transfer_pool(
    allocation,
    host_bytes=4L,
    lanes=1,
    max_regions_per_lane=1,
  )
  errdefer pool.close()
  pool.check_prepared()
  {
    allocation,
    pool,
    payload: FixedArray::make(4, b'\x00'),
    vocabulary: vocabulary_size,
    state: Ready,
  }
}

///|
/// Borrow only during startup binding; close the consuming queue before this
/// input so its allocation's launch-argument leases have been released.
pub fn CommittedTokenSeed::allocation(self : Self) -> @device.Allocation raise {
  guard self.state is Ready else {
    raise @device.DeviceError::Busy(StreamingTransfer)
  }
  self.allocation
}

///|
pub fn CommittedTokenSeed::begin(self : Self, token : Int) -> Unit raise {
  guard self.state is Ready else {
    raise @device.DeviceError::Busy(StreamingTransfer)
  }
  guard token >= 0 && token < self.vocabulary else {
    raise @device.DeviceError::InvalidArgument(StreamingTransfer)
  }
  self.state = Failed
  let bits = token.reinterpret_as_uint()
  for i in 0..<4 {
    self.payload[i] = ((bits >> (i * 8)) & 255U).to_byte()
  }
  self.pool.write_host(self.payload, host_offset=0L, source_offset=0, bytes=4)
  self.pool.enqueue(
    lane=0,
    host_offset=0L,
    device_offset=0L,
    bytes=4L,
    to_device=true,
  )
  self.pool.record(0)
  self.state = Uploading
}

///|
pub fn CommittedTokenSeed::poll(self : Self) -> Bool raise {
  guard self.state is Uploading else {
    raise @device.DeviceError::InvalidArgument(StreamingTransfer)
  }
  self.state = Failed
  if !self.pool.poll(0) {
    self.state = Uploading
    return false
  }
  self.pool.reset(0)
  self.state = Ready
  true
}

///|
pub fn CommittedTokenSeed::close(self : Self) -> Unit raise {
  if self.state is Closed {
    return
  }
  self.state = Failed
  self.pool.drain(0)
  self.pool.close()
  self.allocation.close()
  self.state = Closed
}
