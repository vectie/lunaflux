///|
priv enum State {
  Fresh
  Failed
  Prepared
  Closed
}

///|
/// Owns two intermediates/functions, borrowing inputs and compact weights.
/// Its parent queue owns submission/completion; close that queue first.
pub struct RowProjectionNormFrame {
  priv source : @source.RowProjectionNormCudaSource
  priv buffers : Array[@device.Allocation]
  priv functions : Array[@device.Function]
  priv launches : Array[@device.OrderedKernelLaunch]
  priv mut state : State
}

///|
pub fn RowProjectionNormFrame::new(
  source : @source.RowProjectionNormCudaSource,
) -> Self {
  { source, buffers: [], functions: [], launches: [], state: Fresh, }
}

///|
pub fn RowProjectionNormFrame::source_bytes(self : Self) -> Bytes {
  self.source.source_bytes()
}

///|
pub fn RowProjectionNormFrame::plan(
  self : Self,
) -> @precision.RowProjectionNormPrecision {
  self.source.plan()
}

///|
fn region(
  buffer : @device.Allocation,
  bytes : Int64,
  alignment : Int64,
  offset? : Int64 = 0L,
) -> @device.KernelArgument raise {
  @device.KernelArgument::device_region(
    buffer,
    offset~,
    byte_count=bytes,
    alignment~,
  )
}

///|
pub fn RowProjectionNormFrame::prepare(
  self : Self,
  context : @device.Context,
  aot : @device.Module,
  counts~ : @device.Allocation,
  input~ : @device.Allocation,
  matrix~ : @device.Allocation,
  norm~ : @device.Allocation,
  maximum_workspace_bytes~ : Int64,
) -> Unit raise {
  guard self.state is Fresh else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  let plan = self.plan()
  guard plan.workspace_bytes() <= maximum_workspace_bytes else {
    raise @precision.PrecisionPlanError::MemoryBudgetExceeded
  }
  let (_, width, _) = plan.geometry()
  let rep = plan.weight_representation()
  // Check context ownership as well as all compact spans before allocation.
  for
    (buffer, offset, bytes, alignment) in [
      (counts, 0L, 20L, 4L),
      (input, 0L, plan.input_bytes(), 2L),
      (matrix, 0L, rep.payload_bytes(), 1L),
      (matrix, rep.payload_bytes(), rep.scale_bytes(), 1L),
      (norm, 0L, width.to_int64() * 2L, 2L),
    ] {
    context.validate_allocation_region(
      buffer,
      offset~,
      byte_count=bytes,
      alignment~,
    )
  }
  let count_arg = region(counts, 20L, 4L)
  let input_arg = region(input, plan.input_bytes(), 2L)
  let payload = region(matrix, rep.payload_bytes(), 1L)
  let scales = region(matrix, rep.scale_bytes(), 1L, offset=rep.payload_bytes())
  let norm_arg = region(norm, width.to_int64() * 2L, 2L)
  self.state = Failed
  for _ in 0..<2 {
    self.buffers.push(context.allocate(plan.frame_bytes()))
  }
  let projected = region(self.buffers[0], plan.frame_bytes(), 2L)
  let normalized = region(self.buffers[1], plan.frame_bytes(), 2L)
  let args = [
    [count_arg, input_arg, payload, scales, projected],
    [count_arg, projected, norm_arg, normalized],
  ]
  for stage in 0..<2 {
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
        @device.KernelArguments::new(args[stage]),
      ),
    )
  }
  self.state = Prepared
}

///|
pub fn RowProjectionNormFrame::launches(
  self : Self,
) -> ArrayView[@device.OrderedKernelLaunch] raise {
  guard self.state is Prepared else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.launches[:]
}

///|
pub fn RowProjectionNormFrame::output(self : Self) -> @device.Allocation raise {
  guard self.state is Prepared else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.buffers[1]
}

///|
pub fn RowProjectionNormFrame::close(self : Self) -> Unit raise {
  if self.state is Closed {
    return
  }
  self.state = Failed
  self.launches.clear()
  while self.functions.last() is Some(f) {
    f.close()
    ignore(self.functions.pop())
  }
  while self.buffers.last() is Some(b) {
    b.close()
    ignore(self.buffers.pop())
  }
  self.state = Closed
}
