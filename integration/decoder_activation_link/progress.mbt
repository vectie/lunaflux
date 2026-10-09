///|
fn ActivationLink::write_header(self : ActivationLink) -> Unit {
  for index in 0..<8 {
    self.payload[index] = (self.epoch >> (index * 8)).to_byte()
    self.payload[index + 8] = (self.rows.to_uint64() >> (index * 8)).to_byte()
  }
}

///|
fn ActivationLink::validate_header(
  self : ActivationLink,
  length : Int,
) -> Unit raise {
  guard length == self.rows * self.row_bytes + activation_header_bytes else {
    raise MismatchedStep
  }
  let mut epoch = 0UL
  let mut rows = 0UL
  for index in 0..<8 {
    epoch = epoch | (self.payload[index].to_uint64() << (index * 8))
    rows = rows | (self.payload[index + 8].to_uint64() << (index * 8))
  }
  guard epoch == self.epoch && rows == self.rows.to_uint64() else {
    raise MismatchedStep
  }
}

///|
/// One bounded progress action. Completion means the sender's complete frame
/// left its lane, or the receiver's complete activation reached its device.
/// Socket and DMA events are polled independently, never synchronously waited.
pub async fn ActivationLink::poll(
  self : Self,
  local_now_millis~ : UInt64,
) -> Bool {
  guard !self.busy && self.pool is Some(pool) else { raise InvalidState }
  let state = self.state
  guard state
    is (WaitingProducer | Downloading | Sending | Receiving | Uploading) else {
    raise InvalidState
  }
  self.busy = true
  self.state = Failed
  defer {
    self.busy = false
  }
  match state {
    WaitingProducer => {
      guard self.producer_ready is Some(ready) else { raise InvalidState }
      if ready() {
        pool.enqueue(
          lane=0,
          host_offset=0L,
          device_offset=0L,
          bytes=(self.rows * self.row_bytes).to_int64(),
          to_device=false,
        )
        pool.record(0)
        self.state = Downloading
      } else {
        self.state = WaitingProducer
      }
    }
    Downloading =>
      if pool.poll(0) {
        pool.read_host(
          self.payload,
          host_offset=0L,
          destination_offset=activation_header_bytes,
          bytes=self.rows * self.row_bytes,
        )
        pool.reset(0)
        self.write_header()
        self.channel.begin_send(
          self.payload,
          self.rows * self.row_bytes + activation_header_bytes,
          local_now_millis,
        )
        self.state = Sending
      } else {
        self.state = Downloading
      }
    Sending => {
      if self.channel.poll_send(local_now_millis) {
        self.state = Idle
        return true
      }
      self.state = Sending
    }
    Receiving =>
      match self.channel.poll_receive(self.payload, local_now_millis) {
        None => self.state = Receiving
        Some(length) => {
          self.validate_header(length)
          pool.write_host(
            self.payload,
            host_offset=0L,
            source_offset=activation_header_bytes,
            bytes=self.rows * self.row_bytes,
          )
          pool.enqueue(
            lane=0,
            host_offset=0L,
            device_offset=0L,
            bytes=(self.rows * self.row_bytes).to_int64(),
            to_device=true,
          )
          pool.record(0)
          self.state = Uploading
        }
      }
    Uploading => {
      if pool.poll(0) {
        pool.reset(0)
        self.state = Idle
        return true
      }
      self.state = Uploading
    }
    _ => raise InvalidState
  }
  false
}
