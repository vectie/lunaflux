///|
pub(all) suberror ActivationLinkError {
  InvalidGeometry
  HostBudgetExceeded
  InvalidState
  MismatchedStep
}

///|
priv enum LinkState {
  Fresh
  Idle
  WaitingProducer
  Downloading
  Sending
  Receiving
  Uploading
  Failed
  Closed
}

///|
// Two U64 fields: step epoch and live rows. The existing channel caps complete
// payloads at 64 MiB; this bound is not a model or tile heuristic.
let activation_header_bytes : Int = 16

///|
let channel_frame_limit : Int = 67108864

///|
/// A startup-bound, plain TCP activation edge. The channel and pinned transfer
/// lane are owned; the device allocation and producer callback are borrowed.
/// Consumer work may start only after poll returns true. This is a functional
/// execution dependency, not a diagnostic readback or a TLS admission step.
pub struct ActivationLink {
  priv channel : @channel.RemoteChannel
  priv maximum_rows : Int
  priv row_bytes : Int
  priv payload : FixedArray[Byte]
  priv producer_ready : (() -> Bool raise)?
  priv mut pool : @device.TransferPool?
  priv mut state : LinkState
  priv mut busy : Bool
  priv mut epoch : UInt64
  priv mut rows : Int
}

///|
/// Includes one payload, both channel lanes, and one pinned activation lane.
/// The socket/async runtime and borrowed device allocation are budgeted outside.
pub fn required_host_bytes(
  maximum_rows~ : Int,
  row_bytes~ : Int,
) -> Int64 raise ActivationLinkError {
  guard maximum_rows > 0 &&
    row_bytes > 0 &&
    maximum_rows.to_int64() * row_bytes.to_int64() <=
    (channel_frame_limit - activation_header_bytes).to_int64() else {
    raise InvalidGeometry
  }
  // Three framed heap buffers + one pinned data buffer; two channel prefixes.
  4L * maximum_rows.to_int64() * row_bytes.to_int64() +
  3L * activation_header_bytes.to_int64() +
  8L
}

///|
fn create(
  connection : @socket.Tcp,
  maximum_rows : Int,
  row_bytes : Int,
  maximum_host_bytes : Int64,
  timeout_millis : UInt64,
  producer_ready : (() -> Bool raise)?,
) -> ActivationLink raise {
  guard required_host_bytes(maximum_rows~, row_bytes~) <= maximum_host_bytes else {
    raise HostBudgetExceeded
  }
  let bytes = maximum_rows * row_bytes + activation_header_bytes
  let channel = @channel.RemoteChannel::new(
    connection,
    max_frame_bytes=bytes,
    timeout_millis~,
  )
  {
    channel,
    maximum_rows,
    row_bytes,
    payload: FixedArray::make(bytes, b'\x00'),
    producer_ready,
    pool: None,
    state: Fresh,
    busy: false,
    epoch: 0UL,
    rows: 0,
  }
}

///|
/// Transfers socket ownership on success. The producer is polled before any
/// device read: a pending CUDA queue never races an activation download.
pub fn ActivationLink::sender(
  connection : @socket.Tcp,
  maximum_rows~ : Int,
  row_bytes~ : Int,
  maximum_host_bytes~ : Int64,
  timeout_millis~ : UInt64,
  producer_ready~ : () -> Bool raise,
) -> Self raise {
  create(
    connection,
    maximum_rows,
    row_bytes,
    maximum_host_bytes,
    timeout_millis,
    Some(producer_ready),
  )
}

///|
pub fn ActivationLink::receiver(
  connection : @socket.Tcp,
  maximum_rows~ : Int,
  row_bytes~ : Int,
  maximum_host_bytes~ : Int64,
  timeout_millis~ : UInt64,
) -> Self raise {
  create(
    connection,
    maximum_rows,
    row_bytes,
    maximum_host_bytes,
    timeout_millis,
    None,
  )
}

///|
/// Retain this owner after a failed preparation and explicitly close it.
pub fn ActivationLink::prepare(
  self : Self,
  context : @device.Context,
  activation : @device.Allocation,
) -> Unit raise {
  guard self.state is Fresh &&
    activation.byte_count() >= (self.maximum_rows * self.row_bytes).to_int64() else {
    raise InvalidState
  }
  self.state = Failed
  let pool = context.prepare_transfer_pool(
    activation,
    host_bytes=(self.maximum_rows * self.row_bytes).to_int64(),
    lanes=1,
    max_regions_per_lane=1,
  )
  self.pool = Some(pool)
  pool.check_prepared()
  self.state = Idle
}

///|
pub fn ActivationLink::begin(
  self : Self,
  epoch~ : UInt64,
  live_rows~ : Int,
  local_now_millis~ : UInt64,
) -> Unit raise {
  guard !self.busy &&
    self.state is Idle &&
    epoch > self.epoch &&
    live_rows > 0 &&
    live_rows <= self.maximum_rows else {
    raise InvalidState
  }
  self.epoch = epoch
  self.rows = live_rows
  self.state = Failed
  if self.producer_ready is Some(_) {
    self.state = WaitingProducer
  } else {
    self.channel.begin_receive(local_now_millis)
    self.state = Receiving
  }
}

///|
/// Cancellation/failure drains DMA before releasing the allocation lease.
/// Close the link before closing its borrowed device allocation/context.
pub fn ActivationLink::close(self : Self) -> Unit raise {
  guard !self.busy else { raise InvalidState }
  if self.state is Closed {
    return
  }
  self.state = Failed
  self.channel.close()
  if self.pool is Some(pool) {
    if pool.is_prepared() {
      pool.drain(0)
    }
    pool.close()
    self.pool = None
  }
  self.state = Closed
}
