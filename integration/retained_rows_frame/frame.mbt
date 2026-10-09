///|
priv enum State {
  Fresh
  Prepared
  Failed
  Closed
}

///|
pub struct RetainedRowsFrame {
  priv source : @source.RetainedRowsCudaSource
  priv buffers : Array[@device.Allocation]
  priv functions : Array[@device.Function]
  priv launches : Array[@device.OrderedKernelLaunch]
  priv mut state : State
}

///|
pub fn RetainedRowsFrame::new(source : @source.RetainedRowsCudaSource) -> Self {
  { source, buffers: [], functions: [], launches: [], state: Fresh, }
}

///|
pub fn RetainedRowsFrame::required_device_bytes(self : Self) -> Int64 {
  self.source
  .plan()
  .state_bytes()
  .fold(init=self.source.plan().workspace_bytes(), (total, bytes) => {
    total + bytes
  })
}

///|
pub fn RetainedRowsFrame::source_bytes(self : Self) -> Bytes {
  self.source.source_bytes()
}

///|
fn region(
  allocation : @device.Allocation,
  bytes : Int64,
  alignment : Int64,
) -> @device.KernelArgument raise {
  @device.KernelArgument::device_region(
    allocation,
    offset=0L,
    byte_count=bytes,
    alignment~,
  )
}

///|
/// All input ports are borrowed. Initialize metadata once, never the large
/// cache: rows beyond the published length are not visible to consumers.
pub fn RetainedRowsFrame::prepare(
  self : Self,
  context : @device.Context,
  aot : @device.Module,
  counts~ : @device.Allocation,
  positions~ : @device.Allocation,
  reset_flag~ : @device.Allocation,
  upstream_append~ : @device.Allocation,
  input~ : @device.Allocation,
  maximum_device_bytes~ : Int64,
) -> Unit raise {
  guard self.state is Fresh &&
    self.required_device_bytes() <= maximum_device_bytes else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.state = Failed
  let (width, _, rows, _) = self.source.plan().geometry()
  let c = region(counts, 20L, 4L)
  let p = region(positions, rows.to_int64() * 4L, 4L)
  let r = region(reset_flag, 4L, 4L)
  let e = region(upstream_append, 12L, 4L)
  let x = region(input, rows.to_int64() * width.to_int64() * 2L, 2L)
  let sizes = [
    ..self.source.plan().state_bytes(),
    self.source.plan().workspace_bytes(),
  ]
  for bytes in sizes {
    self.buffers.push(context.allocate(bytes))
  }
  let zero = FixedArray::make(20, b'\x00')
  context.copy_from_fixed_host(self.buffers[1], zero, 0L, 0L, 8L)
  context.copy_from_fixed_host(self.buffers[2], zero, 0L, 0L, 20L)
  let b = self.buffers.mapi((i, buffer) => {
    region(buffer, sizes[i], if i == 0 { 2L } else { 4L })
  })
  let arguments = [
    [c, p, r, e, b[1], b[2], b[3]],
    [b[3], x, b[0]],
    [b[3], b[1], b[2]],
  ]
  for stage in 0..<3 {
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
pub fn RetainedRowsFrame::launches(
  self : Self,
) -> ArrayView[@device.OrderedKernelLaunch] raise {
  guard self.state is Prepared else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.launches[:]
}

///|
fn RetainedRowsFrame::buffer(
  self : Self,
  index : Int,
) -> @device.Allocation raise {
  guard self.state is Prepared else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.buffers[index]
}

///|
pub fn RetainedRowsFrame::values(self : Self) -> @device.Allocation raise {
  self.buffer(0)
}

///|
pub fn RetainedRowsFrame::counts(self : Self) -> @device.Allocation raise {
  self.buffer(2)
}

///|
pub fn RetainedRowsFrame::append_descriptor(
  self : Self,
) -> @device.Allocation raise {
  self.buffer(3)
}

///|
pub fn RetainedRowsFrame::close(self : Self) -> Unit raise {
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
