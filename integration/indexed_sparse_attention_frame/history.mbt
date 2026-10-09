///|
/// One request owns its projected history independently of frame/batch rows.
/// Buffers persist across prefill/decode and are reset by a device flag before
/// reuse. Close borrowing queues/attention frames before closing this owner.
pub struct IndexedSparseHistory {
  priv source : @source.IndexedSparseHistoryCudaSource
  priv buffers : Array[@device.Allocation]
  priv functions : Array[@device.Function]
  priv launches : Array[@device.OrderedKernelLaunch]
  priv mut counts : @device.Allocation?
  priv mut status : State
}

///|
pub fn IndexedSparseHistory::new(
  source : @source.IndexedSparseHistoryCudaSource,
) -> Self {
  {
    source,
    buffers: [],
    functions: [],
    launches: [],
    counts: None,
    status: Fresh,
  }
}

///|
pub fn IndexedSparseHistory::state_bytes(self : Self) -> Int64 {
  self.source.plan().state_bytes()
}

///|
pub fn IndexedSparseHistory::source_bytes(self : Self) -> Bytes {
  self.source.source_bytes()
}

///|
/// Startup only: initialize metadata using bounded host scratch, not a mirror
/// of the retained keys/values. Every append thereafter stays on the device.
fn zero_metadata(
  allocation : @device.Allocation,
) -> Unit raise @device.DeviceError {
  let zeros = Bytes::make(65536, b'\x00')
  for offset = 0L; offset < allocation.byte_count(); {
    let length = (allocation.byte_count() - offset).min(65536L)
    allocation.copy_from_host(zeros, 0L, offset, length)
    continue offset + length
  }
}

///|
pub fn IndexedSparseHistory::prepare(
  self : Self,
  context : @device.Context,
  aot : @device.Module,
  maximum_state_bytes~ : Int64,
  query_counts~ : @device.Allocation,
  reset_flag~ : @device.Allocation,
  index_keys~ : @device.Allocation,
  pool_gate_scores~ : @device.Allocation,
  keys~ : @device.Allocation,
  values~ : @device.Allocation,
) -> Unit raise @device.DeviceError {
  guard self.status is Fresh && self.state_bytes() <= maximum_state_bytes else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.status = Failed
  let sizes = self.source.plan().buffer_bytes()
  for size in sizes {
    self.buffers.push(context.allocate(size))
  }
  zero_metadata(self.buffers[4])
  zero_metadata(self.buffers[5])
  let (rows, history) = self.source.plan().attention().capacity()
  let r = rows.to_int64()
  let h = history.to_int64()
  let buffers = self.buffers
  let arguments = [
    [
      region(query_counts, 20L, 4L),
      region(reset_flag, 4L, 4L),
      region(buffers[5], 20L, 4L),
      region(buffers[4], sizes[4], 4L),
      region(buffers[6], 12L, 4L),
      region(buffers[7], sizes[7], 4L),
    ],
    [
      region(buffers[6], 12L, 4L),
      region(index_keys, sizes[0] / h * r, 2L),
      region(pool_gate_scores, sizes[1] / h * r, 2L),
      region(keys, sizes[2] / h * r, 2L),
      region(values, sizes[3] / h * r, 2L),
      region(buffers[0], sizes[0], 2L),
      region(buffers[1], sizes[1], 2L),
      region(buffers[2], sizes[2], 2L),
      region(buffers[3], sizes[3], 2L),
    ],
    [
      region(buffers[6], 12L, 4L),
      region(buffers[5], 20L, 4L),
      region(buffers[4], sizes[4], 4L),
    ],
  ]
  for stage in 0..<3 {
    let (grid_x, block_x) = self.source.launch(stage)
    let function = aot.load_function(@utf8.encode(self.source.symbol(stage)))
    self.functions.push(function)
    self.launches.push(
      @device.OrderedKernelLaunch::new(
        function,
        @device.KernelLaunchDimensions::new(
          grid_x~,
          grid_y=1,
          grid_z=1,
          block_x~,
          block_y=1,
          block_z=1,
          shared_memory_bytes=0,
        ),
        @device.KernelArguments::new(arguments[stage]),
      ),
    )
  }
  self.status = Prepared
  self.counts = Some(query_counts)
}

///|
pub fn IndexedSparseHistory::launches(
  self : Self,
) -> ArrayView[@device.OrderedKernelLaunch] raise @device.DeviceError {
  guard self.status is Prepared else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.launches[:]
}

///|
/// I32 [base, live rows, capacity/metadata error]. Borrow for device-side error
/// propagation; diagnostic readback is allowed only after queue completion.
pub fn IndexedSparseHistory::append_descriptor(
  self : Self,
) -> @device.Allocation raise @device.DeviceError {
  guard self.status is Prepared else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.buffers[6]
}

///|
/// Device positions assigned by the reserve launch. A complete decoder may
/// place its projection/rotary launches between reserve and copy/publication,
/// borrowing this region rather than recomputing positions on the host.
pub fn IndexedSparseHistory::query_positions(
  self : Self,
) -> @device.Allocation raise @device.DeviceError {
  guard self.status is Prepared else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.buffers[7]
}

///|
pub fn IndexedSparseHistory::close(
  self : Self,
) -> Unit raise @device.DeviceError {
  if self.status is Closed {
    return
  }
  self.status = Failed
  self.launches.clear()
  self.counts = None
  while self.functions.last() is Some(function) {
    function.close()
    ignore(self.functions.pop())
  }
  while self.buffers.last() is Some(buffer) {
    buffer.close()
    ignore(self.buffers.pop())
  }
  self.status = Closed
}
