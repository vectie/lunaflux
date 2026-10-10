///|
priv enum Phase {
  Ready
  Reading
  Retired
  Failed
  Closed
}

///|
/// One prepared output transfer. Prediction must retire before begin(); this
/// owner does not poll the model, alter KV or perform a diagnostic round trip.
/// It leases the source, so close it before the predictor's output allocation.
pub struct DraftTokenReadback {
  priv pool : @device.TransferPool
  priv payload : FixedArray[Byte]
  priv count : Int
  priv vocabulary : Int
  priv mut phase : Phase
}

///|
pub fn DraftTokenReadback::new(
  context : @device.Context,
  source : @device.Allocation,
  token_count~ : Int,
  vocabulary_size~ : Int,
  maximum_host_bytes~ : Int64,
) -> Self raise {
  guard token_count > 0 &&
    token_count <= 536870911 &&
    vocabulary_size > 0 &&
    token_count.to_int64() * 4L <= source.byte_count() &&
    token_count.to_int64() * 4L <= maximum_host_bytes else {
    raise @device.DeviceError::InvalidArgument(StreamingTransfer)
  }
  let pool = context.prepare_transfer_pool(
    source,
    host_bytes=token_count.to_int64() * 4L,
    lanes=1,
    max_regions_per_lane=1,
  )
  errdefer pool.close()
  pool.check_prepared()
  {
    pool,
    payload: FixedArray::make(token_count * 4, b'\x00'),
    count: token_count,
    vocabulary: vocabulary_size,
    phase: Ready,
  }
}

///|
pub fn DraftTokenReadback::begin(self : Self) -> Unit raise {
  guard self.phase is (Ready | Retired) else {
    raise @device.DeviceError::Busy(StreamingTransfer)
  }
  self.phase = Failed
  self.pool.enqueue(
    lane=0,
    host_offset=0L,
    device_offset=0L,
    bytes=self.count.to_int64() * 4L,
    to_device=false,
  )
  self.pool.record(0)
  self.phase = Reading
}

///|
fn DraftTokenReadback::sample(self : Self, row : Int) -> Int {
  for i in 0..<4; bits = 0U {
    continue bits | (self.payload[row * 4 + i].to_uint() << (i * 8))
  } nobreak {
    bits.reinterpret_as_int()
  }
}

///|
pub fn DraftTokenReadback::poll(self : Self) -> Bool raise {
  guard self.phase is Reading else {
    raise @device.DeviceError::InvalidArgument(StreamingTransfer)
  }
  self.phase = Failed
  if !self.pool.poll(0) {
    self.phase = Reading
    return false
  }
  self.pool.read_host(
    self.payload,
    host_offset=0L,
    destination_offset=0,
    bytes=self.payload.length(),
  )
  self.pool.reset(0)
  for row in 0..<self.count {
    let token = self.sample(row)
    guard token >= 0 && token < self.vocabulary else {
      raise @device.DeviceError::InvalidOutput(StreamingTransfer)
    }
  }
  self.phase = Retired
  true
}

///|
pub fn DraftTokenReadback::token_count(self : Self) -> Int raise {
  guard self.phase is Retired else {
    raise @device.DeviceError::Busy(StreamingTransfer)
  }
  self.count
}

///|
pub fn DraftTokenReadback::token_at(self : Self, row : Int) -> Int raise {
  guard self.phase is Retired && row >= 0 && row < self.count else {
    raise @device.DeviceError::Busy(StreamingTransfer)
  }
  self.sample(row)
}

///|
pub fn DraftTokenReadback::close(self : Self) -> Unit raise {
  if self.phase is Closed {
    return
  }
  self.phase = Failed
  self.pool.drain(0)
  self.pool.close()
  self.phase = Closed
}
