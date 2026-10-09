///|
priv enum State {
  Fresh
  Ready
  Active
  Failed
  Closed
}

///|
/// Owns one layer's recurrent cache and AOT queue. All frame operands and the
/// stream/module are borrowed. Request-slot IDs survive batch reordering;
/// resetting a slot begins a new request, not a new batch.
pub struct RecurrentDeltaProgram {
  priv source : @source.RecurrentDeltaCudaSource
  priv maximum_rows : Int
  priv mut status : State
  priv mut cache : @device.Allocation?
  priv mut function : @device.Function?
  priv mut queue : @device.OrderedKernelExecutor?
}

///|
pub fn RecurrentDeltaProgram::new(
  source : @source.RecurrentDeltaCudaSource,
) -> Self {
  let (maximum_rows, _, _) = source.plan().capacity()
  {
    source,
    maximum_rows,
    status: Fresh,
    cache: None,
    function: None,
    queue: None,
  }
}

///|
pub fn RecurrentDeltaProgram::state_bytes(self : Self) -> Int64 {
  self.source.plan().state_bytes()
}

///|
pub fn RecurrentDeltaProgram::source_bytes(self : Self) -> Bytes {
  self.source.source_bytes()
}

///|
fn region(
  allocation : @device.Allocation,
  bytes : Int64,
  alignment : Int64,
) -> @device.KernelArgument raise @device.DeviceError {
  @device.KernelArgument::device_region(
    allocation,
    offset=0L,
    byte_count=bytes,
    alignment~,
  )
}

///|
/// Prepare once. CSR offsets cover live rows; slots are exclusive within each
/// frame and reset flags mark new requests. Metadata may be updated in-place
/// only while idle. State initialization uses bounded startup scratch.
pub fn RecurrentDeltaProgram::prepare(
  self : Self,
  context : @device.Context,
  aot : @device.Module,
  stream : @device.Stream,
  maximum_state_bytes~ : Int64,
  counts~ : @device.Allocation,
  sequence_offsets~ : @device.Allocation,
  sequence_slots~ : @device.Allocation,
  reset_flags~ : @device.Allocation,
  query~ : @device.Allocation,
  key~ : @device.Allocation,
  value~ : @device.Allocation,
  log_decay~ : @device.Allocation,
  beta~ : @device.Allocation,
  output~ : @device.Allocation,
) -> Unit raise @device.DeviceError {
  guard self.status is Fresh &&
    self.cache is None &&
    self.function is None &&
    self.state_bytes() <= maximum_state_bytes else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  let plan = self.source.plan()
  let (heads, keys, values) = plan.geometry()
  let (rows, sequences, _) = plan.capacity()
  let qbytes = rows.to_int64() * heads.to_int64() * keys.to_int64() * 2L
  let vbytes = rows.to_int64() * heads.to_int64() * values.to_int64() * 2L
  let state = context.allocate(self.state_bytes())
  self.cache = Some(state)
  let zero = FixedArray::make(65536, b'\x00')
  for offset = 0L; offset < self.state_bytes(); {
    let remaining = self.state_bytes() - offset
    let bytes = if remaining < 65536L { remaining } else { 65536L }
    context.copy_from_fixed_host(state, zero, 0L, offset, bytes)
    continue offset + bytes
  }
  let function = aot.load_function(@utf8.encode(self.source.symbol()))
  self.function = Some(function)
  let (grid_x, grid_y, block_x) = self.source.launch()
  let launch = @device.OrderedKernelLaunch::new(
    function,
    @device.KernelLaunchDimensions::new(
      grid_x~,
      grid_y~,
      grid_z=1,
      block_x~,
      block_y=1,
      block_z=1,
      shared_memory_bytes=0,
    ),
    @device.KernelArguments::new([
      region(counts, 20L, 4L),
      region(sequence_offsets, (sequences.to_int64() + 1L) * 4L, 4L),
      region(sequence_slots, sequences.to_int64() * 4L, 4L),
      region(reset_flags, sequences.to_int64() * 4L, 4L),
      region(query, qbytes, 2L),
      region(key, qbytes, 2L),
      region(value, vbytes, 2L),
      region(log_decay, qbytes * 2L, 4L),
      region(beta, rows.to_int64() * heads.to_int64() * 2L, 2L),
      region(state, self.state_bytes(), 4L),
      region(output, vbytes, 2L),
    ]),
  )
  self.queue = Some(context.create_ordered_kernel_executor(stream, [launch]))
  self.status = Ready
}

///|
pub fn RecurrentDeltaProgram::submit(
  self : Self,
  live_rows~ : Int,
) -> Unit raise @device.DeviceError {
  guard self.status is Ready && self.queue is Some(queue) else {
    raise @device.DeviceError::Busy(OrderedExecutorEnqueue)
  }
  guard live_rows > 0 && live_rows <= self.maximum_rows else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorEnqueue)
  }
  errdefer {
    self.status = Failed
  }
  queue.enqueue(0)
  queue.record_completion()
  self.status = Active
}

///|
pub fn RecurrentDeltaProgram::poll(
  self : Self,
) -> Bool raise @device.DeviceError {
  guard self.status is Active && self.queue is Some(queue) else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorPoll)
  }
  errdefer {
    self.status = Failed
  }
  if queue.poll_completion() {
    queue.reset()
    self.status = Ready
    true
  } else {
    false
  }
}

///|
pub fn RecurrentDeltaProgram::close(
  self : Self,
) -> Unit raise @device.DeviceError {
  if self.status is Closed {
    return
  }
  guard self.status is (Fresh | Ready) else {
    raise @device.DeviceError::Busy(OrderedExecutorClose)
  }
  if self.queue is Some(queue) {
    queue.close()
    self.queue = None
  }
  if self.function is Some(function) {
    function.close()
    self.function = None
  }
  if self.cache is Some(cache) {
    cache.close()
    self.cache = None
  }
  self.status = Closed
}

///|
/// Cancellation drains before releasing state. Partially updated request
/// state is never reused after abort: this owner becomes terminal.
pub fn RecurrentDeltaProgram::abort(
  self : Self,
) -> Unit raise @device.DeviceError {
  if self.status is Closed {
    return
  }
  if self.queue is Some(queue) {
    self.status = Failed
    queue.abort()
  }
  self.status = Fresh
  self.close()
}
