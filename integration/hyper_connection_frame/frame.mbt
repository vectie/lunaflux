///|
priv enum State {
  Fresh
  Prepared
  Failed
  Closed
}

///|
/// Owns the complete branch envelope's frames/functions. Its launch views are
/// borrowed into the branch's single queue and must outlive that queue.
pub struct HyperConnectionFrame {
  priv source : @source.HyperConnectionCudaSource
  priv buffers : Array[@device.Allocation]
  priv functions : Array[@device.Function]
  priv before : Array[@device.OrderedKernelLaunch]
  priv after : Array[@device.OrderedKernelLaunch]
  priv mut status : State
}

///|
pub fn HyperConnectionFrame::new(
  source : @source.HyperConnectionCudaSource,
) -> Self {
  { source, buffers: [], functions: [], before: [], after: [], status: Fresh, }
}

///|
pub fn HyperConnectionFrame::source_bytes(self : Self) -> Bytes {
  self.source.source_bytes()
}

///|
pub fn HyperConnectionFrame::workspace_bytes(self : Self) -> Int64 {
  self.source.plan().workspace_bytes()
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
fn HyperConnectionFrame::launch(
  self : Self,
  aot : @device.Module,
  stage : Int,
  grid : Int,
  threads : Int,
  arguments : Array[@device.KernelArgument],
) -> @device.OrderedKernelLaunch raise {
  let function = aot.load_function(@utf8.encode(self.source.symbol(stage)))
  self.functions.push(function)
  @device.OrderedKernelLaunch::new(
    function,
    @device.KernelLaunchDimensions::new(
      grid_x=grid,
      grid_y=1,
      grid_z=1,
      block_x=threads,
      block_y=1,
      block_z=1,
      shared_memory_bytes=0,
    ),
    @device.KernelArguments::new(arguments),
  )
}

///|
pub fn HyperConnectionFrame::prepare(
  self : Self,
  context : @device.Context,
  aot : @device.Module,
  counts~ : @device.Allocation,
  input~ : @device.Allocation,
  output~ : @device.Allocation,
  function_weight~ : @device.Allocation,
  base~ : @device.Allocation,
  scale~ : @device.Allocation,
  norm_weight~ : @device.Allocation,
  maximum_workspace_bytes~ : Int64,
) -> Unit raise {
  guard self.status is Fresh &&
    self.workspace_bytes() <= maximum_workspace_bytes &&
    function_weight.byte_count() >= self.source.plan().function_weight_bytes() else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.status = Failed
  let plan = self.source.plan()
  let (streams, hidden, rows) = plan.geometry()
  for size in plan.frame_bytes() {
    self.buffers.push(context.allocate(size))
  }
  let frame = self.buffers
  let sizes = plan.frame_bytes()
  let stream_bytes = rows.to_int64() *
    streams.to_int64() *
    hidden.to_int64() *
    2L
  let width = (streams.to_int64() + 2L) * streams.to_int64()
  self.before.push(
    self.launch(aot, 0, rows, 1, [
      region(counts, 20L, 4L),
      region(input, stream_bytes, 2L),
      region(
        function_weight,
        plan.function_weight_bytes(),
        match plan.control_law() {
          NormalizedBf16Projection => 2L
          F32ProjectionThenNormalize => 4L
        },
      ),
      region(base, width * 4L, 4L),
      region(scale, 12L, 4L),
      region(frame[0], sizes[0], 4L),
      region(frame[1], sizes[1], 4L),
      region(frame[3], sizes[3], 2L),
    ]),
  )
  self.before.push(
    self.launch(aot, 1, rows, 1, [
      region(counts, 20L, 4L),
      region(frame[1], sizes[1], 4L),
      region(frame[2], sizes[2], 4L),
    ]),
  )
  self.before.push(
    self.launch(aot, 2, rows, 256, [
      region(counts, 20L, 4L),
      region(frame[3], sizes[3], 2L),
      region(norm_weight, hidden.to_int64() * 2L, 2L),
      region(frame[4], sizes[4], 2L),
    ]),
  )
  let post = region(frame[0], sizes[0], 4L)
  let combination = region(frame[2], sizes[2], 4L)
  let residual = region(input, stream_bytes, 2L)
  let branch = region(frame[5], sizes[5], 2L)
  let arguments = match plan.mix_law() {
    RoundedBf16ControlsAndProducts =>
      [
        region(counts, 20L, 4L),
        post,
        combination,
        residual,
        branch,
        region(output, stream_bytes, 2L),
      ]
    OrderedF32SingleBf16Round | TransposedF32ResidualThenBranch =>
      [
        region(counts, 20L, 4L),
        branch,
        residual,
        post,
        combination,
        region(output, stream_bytes, 2L),
      ]
  }
  let (grid, threads) = match plan.mix_law() {
    RoundedBf16ControlsAndProducts => ((stream_bytes / 2L + 255L) / 256L, 256)
    OrderedF32SingleBf16Round | TransposedF32ResidualThenBranch =>
      (rows.to_int64(), 256)
  }
  guard grid <= 2147483647L else {
    raise @precision.PrecisionPlanError::NoImplementation
  }
  self.after.push(self.launch(aot, 3, grid.to_int(), threads, arguments))
  self.status = Prepared
}

///|
pub fn HyperConnectionFrame::prefix(
  self : Self,
) -> ArrayView[@device.OrderedKernelLaunch] raise @device.DeviceError {
  guard self.status is Prepared else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.before[:]
}

///|
pub fn HyperConnectionFrame::suffix(
  self : Self,
) -> ArrayView[@device.OrderedKernelLaunch] raise @device.DeviceError {
  guard self.status is Prepared else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.after[:]
}

///|
pub fn HyperConnectionFrame::branch_input(
  self : Self,
) -> @device.Allocation raise @device.DeviceError {
  guard self.status is Prepared else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.buffers[4]
}

///|
pub fn HyperConnectionFrame::branch_output(
  self : Self,
) -> @device.Allocation raise @device.DeviceError {
  guard self.status is Prepared else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.buffers[5]
}

///|
/// The borrowing queue must be closed before closing this frame owner.
pub fn HyperConnectionFrame::close(
  self : Self,
) -> Unit raise @device.DeviceError {
  match self.status {
    Closed => return
    Fresh | Prepared | Failed => ()
  }
  self.status = Failed
  self.before.clear()
  self.after.clear()
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
