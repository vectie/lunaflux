///|
priv enum State {
  Fresh
  Prepared
  Failed
  Closed
}

///|
/// Owns one request's persistent ring and output. All four operations enter
/// the caller's queue: reserve, attend against old/current KV, copy, publish.
/// There is no private queue or per-token host allocation/readback.
pub struct WindowSharedKvFrame {
  priv source : @source.WindowSharedKvCudaSource
  priv buffers : Array[@device.Allocation]
  priv functions : Array[@device.Function]
  priv launches : Array[@device.OrderedKernelLaunch]
  priv mut state : State
}

///|
pub fn WindowSharedKvFrame::new(
  source : @source.WindowSharedKvCudaSource,
) -> Self {
  { source, buffers: [], functions: [], launches: [], state: Fresh, }
}

///|
pub fn WindowSharedKvFrame::source_bytes(self : Self) -> Bytes {
  self.source.source_bytes()
}

///|
pub fn WindowSharedKvFrame::state_bytes(self : Self) -> Int64 {
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
pub fn WindowSharedKvFrame::prepare(
  self : Self,
  context : @device.Context,
  aot : @device.Module,
  counts~ : @device.Allocation,
  reset_flag~ : @device.Allocation,
  positions~ : @device.Allocation,
  queries~ : @device.Allocation,
  shared_kv~ : @device.Allocation,
  attention_sink~ : @device.Allocation,
  maximum_state_bytes~ : Int64,
) -> Unit raise {
  guard self.state is Fresh && self.state_bytes() <= maximum_state_bytes else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.state = Failed
  let (heads, width, _, rows, _) = self.source.plan().geometry()
  // Validate borrowed spans before acquiring any private device storage.
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
  let sizes = self.source.plan().buffer_bytes()
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
    [count, reset, position, history, append],
    [append, query, kv, ring, sink, output],
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
pub fn WindowSharedKvFrame::launches(
  self : Self,
) -> ArrayView[@device.OrderedKernelLaunch] raise {
  guard self.state is Prepared else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.launches[:]
}

///|
pub fn WindowSharedKvFrame::output(self : Self) -> @device.Allocation raise {
  guard self.state is Prepared else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.buffers[3]
}

///|
/// I32 [base,live,error] is borrowed by the containing decoder's existing
/// error/completion boundary. It must not be polled on the host per layer.
pub fn WindowSharedKvFrame::append_descriptor(
  self : Self,
) -> @device.Allocation raise {
  guard self.state is Prepared else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.buffers[2]
}

///|
pub fn WindowSharedKvFrame::close(self : Self) -> Unit raise {
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
