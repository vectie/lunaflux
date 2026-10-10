///|
priv enum State {
  Fresh
  Prepared
  Failed
  Closed
}

///|
/// Owns the segmented capture buffer and function. Counts and residuals are
/// borrowed; the parent ordered queue must be closed before this frame.
pub struct StreamMeanCaptureFrame {
  priv source : @source.StreamMeanCaptureCudaSource
  priv buffers : Array[@device.Allocation]
  priv functions : Array[@device.Function]
  priv mut counts : @device.Allocation?
  priv mut state : State
}

///|
pub fn StreamMeanCaptureFrame::new(
  source : @source.StreamMeanCaptureCudaSource,
) -> Self {
  { source, buffers: [], functions: [], counts: None, state: Fresh, }
}

///|
pub fn StreamMeanCaptureFrame::source_bytes(self : Self) -> Bytes {
  self.source.source_bytes()
}

///|
pub fn StreamMeanCaptureFrame::plan(
  self : Self,
) -> @precision.StreamMeanCapturePrecision {
  self.source.plan()
}

///|
pub fn StreamMeanCaptureFrame::prepare(
  self : Self,
  context : @device.Context,
  aot : @device.Module,
  counts~ : @device.Allocation,
  maximum_device_bytes~ : Int64,
) -> Unit raise {
  guard self.state is Fresh else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  guard self.plan().output_bytes() <= maximum_device_bytes else {
    raise @precision.PrecisionPlanError::MemoryBudgetExceeded
  }
  context.validate_allocation_region(
    counts,
    offset=0L,
    byte_count=20L,
    alignment=4L,
  )
  self.state = Failed
  self.buffers.push(context.allocate(self.plan().output_bytes()))
  let (_, _, _, segments) = self.plan().geometry()
  for segment in 0..<segments {
    self.functions.push(
      aot.load_function(@utf8.encode(self.source.symbol(segment))),
    )
  }
  self.counts = Some(counts)
  self.state = Prepared
}

///|
/// Binding is startup-only. The resulting launch runs immediately after its
/// producer and before any reusable inter-layer residual can be overwritten.
pub fn StreamMeanCaptureFrame::capture(
  self : Self,
  input : @device.Allocation,
  segment~ : Int,
) -> @device.OrderedKernelLaunch raise {
  let (_, _, _, segments) = self.plan().geometry()
  guard self.state is Prepared &&
    segment >= 0 &&
    segment < segments &&
    self.counts is Some(counts) else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  guard !input.same_storage(self.buffers[0]) else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  @device.OrderedKernelLaunch::new(
    self.functions[segment],
    @device.KernelLaunchDimensions::new(
      grid_x=self.source.grid_x(),
      grid_y=1,
      grid_z=1,
      block_x=256,
      block_y=1,
      block_z=1,
      shared_memory_bytes=0,
    ),
    @device.KernelArguments::new([
      @device.KernelArgument::device_region(
        counts,
        offset=0L,
        byte_count=20L,
        alignment=4L,
      ),
      @device.KernelArgument::device_region(
        input,
        offset=0L,
        byte_count=self.plan().input_bytes(),
        alignment=2L,
      ),
      @device.KernelArgument::device_region(
        self.buffers[0],
        offset=0L,
        byte_count=self.plan().output_bytes(),
        alignment=2L,
      ),
    ]),
  )
}

///|
pub fn StreamMeanCaptureFrame::output(self : Self) -> @device.Allocation raise {
  guard self.state is Prepared else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.buffers[0]
}

///|
pub fn StreamMeanCaptureFrame::close(self : Self) -> Unit raise {
  if self.state is Closed {
    return
  }
  self.state = Failed
  while self.functions.last() is Some(f) {
    f.close()
    ignore(self.functions.pop())
  }
  while self.buffers.last() is Some(b) {
    b.close()
    ignore(self.buffers.pop())
  }
  self.counts = None
  self.state = Closed
}
