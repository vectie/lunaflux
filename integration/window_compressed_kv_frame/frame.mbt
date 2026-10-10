///|
pub(all) enum CompressedReadPorts {
  Learned(
    counts~ : @device.Allocation,
    rows~ : @device.Allocation,
    offsets~ : @device.Allocation
  )
  AllCausal
}

///|
priv enum State {
  Fresh
  Prepared
  Failed
  Closed
}

///|
pub struct WindowCompressedKvFrame {
  priv source : @source.WindowCompressedKvCudaSource
  priv buffers : Array[@device.Allocation]
  priv functions : Array[@device.Function]
  priv launches : Array[@device.OrderedKernelLaunch]
  priv mut state : State
}

///|
pub fn WindowCompressedKvFrame::new(
  source : @source.WindowCompressedKvCudaSource,
) -> Self {
  { source, buffers: [], functions: [], launches: [], state: Fresh, }
}

///|
pub fn WindowCompressedKvFrame::source_bytes(self : Self) -> Bytes {
  self.source.source_bytes()
}

///|
pub fn WindowCompressedKvFrame::state_bytes(self : Self) -> Int64 {
  self.source.plan().state_bytes()
}

///|
fn region(
  buffer : @device.Allocation,
  bytes : Int64,
  alignment : Int64,
) -> @device.KernelArgument raise {
  @device.KernelArgument::device_region(
    buffer,
    offset=0L,
    byte_count=bytes,
    alignment~,
  )
}

///|
pub fn WindowCompressedKvFrame::prepare(
  self : Self,
  context : @device.Context,
  aot : @device.Module,
  counts~ : @device.Allocation,
  reset_flag~ : @device.Allocation,
  positions~ : @device.Allocation,
  queries~ : @device.Allocation,
  shared_kv~ : @device.Allocation,
  attention_sink~ : @device.Allocation,
  compressed_values~ : @device.Allocation,
  compressed_counts~ : @device.Allocation,
  compressed_append~ : @device.Allocation,
  read_ports~ : CompressedReadPorts,
  maximum_state_bytes~ : Int64,
) -> Unit raise {
  guard self.state is Fresh && self.state_bytes() <= maximum_state_bytes else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.state = Failed
  let plan = self.source.plan()
  let (heads, width, _, rows, _) = plan.window().geometry()
  // Every borrowed port is checked before acquiring private storage.
  let count = region(counts, 20L, 4L)
  let reset = region(reset_flag, 4L, 4L)
  let position = region(positions, rows.to_int64() * 4L, 4L)
  let query = region(
    queries,
    rows.to_int64() * heads.to_int64() * width.to_int64() * 2L,
    2L,
  )
  let kv = region(shared_kv, rows.to_int64() * width.to_int64() * 2L, 2L)
  let sink = region(attention_sink, heads.to_int64() * 4L, 4L)
  let compressed = region(
    compressed_values,
    Int::max(plan.cache_capacity(), 1).to_int64() * width.to_int64() * 2L,
    2L,
  )
  let cache_count = region(compressed_counts, 20L, 4L)
  let cache_append = region(compressed_append, 12L, 4L)
  let (selected_count, selected, offset) = match (plan.read_set(), read_ports) {
    (
      @precision.CompressedRowReadSet::Learned(top_k=_),
      Learned(counts~, rows=ids, offsets~),
    ) =>
      (
        region(counts, rows.to_int64() * 4L, 4L),
        region(
          ids,
          rows.to_int64() * Int::max(plan.selected_width(), 1).to_int64() * 4L,
          4L,
        ),
        region(offsets, rows.to_int64() * 4L, 4L),
      )
    (@precision.CompressedRowReadSet::AllCausal, AllCausal) =>
      // Those ABI slots are never dereferenced by the all-causal lowering.
      // Borrow one existing valid span instead of allocating dummy tensors.
      (reset, reset, reset)
    _ => raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  let sizes = plan.window().buffer_bytes()
  for size in sizes {
    self.buffers.push(context.allocate(size))
  }
  self.buffers[1].copy_from_host(
    b"\x00\x00\x00\x00\x00\x00\x00\x00", 0L, 0L, 8L,
  )
  let ring = region(self.buffers[0], sizes[0], 2L)
  let history = region(self.buffers[1], sizes[1], 4L)
  let append = region(self.buffers[2], sizes[2], 4L)
  let output = region(self.buffers[3], sizes[3], 2L)
  let arguments = [
    [
      count, reset, position, cache_count, cache_append, selected_count, selected,
      offset, history, append,
    ],
    [
      append, selected_count, selected, offset, query, kv, ring, compressed, sink,
      output,
    ],
    [append, kv, ring],
    [append, history],
  ]
  for stage in 0..<4 {
    let function = aot.load_function(@utf8.encode(self.source.symbol(stage)))
    self.functions.push(function)
    let (x, y, threads) = self.source.launch(stage)
    self.launches.push(
      @device.OrderedKernelLaunch::new(
        function,
        @device.KernelLaunchDimensions::new(
          grid_x=x,
          grid_y=y,
          grid_z=1,
          block_x=threads,
          block_y=1,
          block_z=1,
          shared_memory_bytes=0,
        ),
        @device.KernelArguments::new(arguments[stage]),
      ),
    )
  }
  self.state = Prepared
}

///|
pub fn WindowCompressedKvFrame::launches(
  self : Self,
) -> ArrayView[@device.OrderedKernelLaunch] raise {
  guard self.state is Prepared else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.launches[:]
}

///|
pub fn WindowCompressedKvFrame::output(self : Self) -> @device.Allocation raise {
  guard self.state is Prepared else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.buffers[3]
}

///|
pub fn WindowCompressedKvFrame::append_descriptor(
  self : Self,
) -> @device.Allocation raise {
  guard self.state is Prepared else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.buffers[2]
}

///|
pub fn WindowCompressedKvFrame::close(self : Self) -> Unit raise {
  if self.state is Closed {
    return
  }
  self.state = Failed
  self.launches.clear()
  while self.functions.last() is Some(function) {
    function.close()
    ignore(self.functions.pop())
  }
  while self.buffers.last() is Some(buffer) {
    buffer.close()
    ignore(self.buffers.pop())
  }
  self.state = Closed
}
