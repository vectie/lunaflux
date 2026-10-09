///|
priv enum State {
  Fresh
  Prepared
  Failed
  Closed
}

///|
/// Persistent learned-pooling state and compact outputs; the containing decoder
/// owns queue submission and borrows the append error boundary.
pub struct LearnedGatedPoolFrame {
  priv source : @source.LearnedGatedPoolCudaSource
  priv buffers : Array[@device.Allocation]
  priv functions : Array[@device.Function]
  priv launches : Array[@device.OrderedKernelLaunch]
  priv mut state : State
}

///|
pub fn LearnedGatedPoolFrame::new(
  source : @source.LearnedGatedPoolCudaSource,
) -> Self {
  { source, buffers: [], functions: [], launches: [], state: Fresh, }
}

///|
pub fn LearnedGatedPoolFrame::required_device_bytes(self : Self) -> Int64 {
  self.source.plan().state_bytes().fold(init=0L, (n, b) => n + b) +
  self.source.plan().frame_bytes().fold(init=0L, (n, b) => n + b)
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
fn initialize(
  context : @device.Context,
  buffer : @device.Allocation,
  bytes : Int64,
  pattern : FixedArray[Byte],
) -> Unit raise {
  // Startup-only bounded host scratch. Every range is four-byte aligned.
  for offset = 0L; offset < bytes; {
    let count = (bytes - offset).min(pattern.length().to_int64())
    context.copy_from_fixed_host(buffer, pattern, 0L, offset, count)
    continue offset + count
  }
}

///|
pub fn LearnedGatedPoolFrame::prepare(
  self : Self,
  context : @device.Context,
  aot : @device.Module,
  counts~ : @device.Allocation,
  positions~ : @device.Allocation,
  reset_flag~ : @device.Allocation,
  input~ : @device.Allocation,
  weights~ : ArrayView[@device.Allocation],
  maximum_device_bytes~ : Int64,
) -> Unit raise {
  guard self.state is Fresh &&
    weights.length() == 4 &&
    self.required_device_bytes() <= maximum_device_bytes else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.state = Failed
  let plan = self.source.plan()
  let (hidden, _, _, _, rows, _) = plan.geometry()
  let c = region(counts, 20L, 4L)
  let p = region(positions, rows.to_int64() * 4L, 4L)
  let r = region(reset_flag, 4L, 4L)
  let x = region(input, rows.to_int64() * hidden.to_int64() * 2L, 2L)
  let w = plan
    .parameter_bytes()
    .mapi((i, bytes) => region(weights[i], bytes, if i == 2 { 4L } else { 2L }))
  let sizes = [..plan.state_bytes(), ..plan.frame_bytes()]
  for bytes in sizes {
    self.buffers.push(context.allocate(bytes))
  }
  let zero = FixedArray::make(65536, b'\x00')
  let negative_infinity = FixedArray::makei(65536, i => {
    match i % 4 {
      2 => b'\x80'
      3 => b'\xff'
      _ => b'\x00'
    }
  })
  initialize(context, self.buffers[0], sizes[0], zero)
  initialize(context, self.buffers[1], sizes[1], negative_infinity)
  initialize(context, self.buffers[2], 8L, zero)
  let b = self.buffers.mapi((i, allocation) => {
    region(allocation, sizes[i], if i == 5 || i == 6 { 2L } else { 4L })
  })
  let arguments = [
    [c, p, r, b[2], b[9], b[8], b[7]],
    [b[9], x, w[0], w[1], b[3], b[4]],
    [b[9], b[3], b[4], w[2], b[0], b[1], b[5]],
    [b[9], b[5], w[3], b[6]],
    [b[9], b[2]],
  ]
  for stage in 0..<5 {
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
pub fn LearnedGatedPoolFrame::launches(
  self : Self,
) -> ArrayView[@device.OrderedKernelLaunch] raise {
  guard self.state is Prepared else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.launches[:]
}

///|
fn LearnedGatedPoolFrame::buffer(
  self : LearnedGatedPoolFrame,
  index : Int,
) -> @device.Allocation raise {
  guard self.state is Prepared else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.buffers[index]
}

///|
pub fn LearnedGatedPoolFrame::output(self : Self) -> @device.Allocation raise {
  self.buffer(6)
}

///|
pub fn LearnedGatedPoolFrame::output_counts(
  self : Self,
) -> @device.Allocation raise {
  self.buffer(8)
}

///|
pub fn LearnedGatedPoolFrame::output_positions(
  self : Self,
) -> @device.Allocation raise {
  self.buffer(7)
}

///|
pub fn LearnedGatedPoolFrame::append_descriptor(
  self : Self,
) -> @device.Allocation raise {
  self.buffer(9)
}

///|
pub fn LearnedGatedPoolFrame::close(self : Self) -> Unit raise {
  if self.state is Closed {
    return
  }
  self.state = Failed
  self.launches.clear()
  while self.functions.last() is Some(function) {
    function.close()
    ignore(self.functions.pop())
  }
  while self.buffers.last() is Some(allocation) {
    allocation.close()
    ignore(self.buffers.pop())
  }
  self.state = Closed
}
