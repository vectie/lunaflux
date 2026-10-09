///|
priv enum State {
  Fresh
  Prepared
  Failed
  Closed
}

///|
/// Prepared ingress/egress around a borrowed decoder frame. The caller owns
/// its queue and closes it before releasing this frame, weights or residuals.
pub struct TextIoFrame {
  priv source : @source.TextIoCudaSource
  priv boundary : @precision.TextBoundary
  priv buffers : Array[@device.Allocation]
  priv functions : Array[@device.Function]
  priv before : Array[@device.OrderedKernelLaunch]
  priv after : Array[@device.OrderedKernelLaunch]
  priv mut state : State
}

///|
pub fn TextIoFrame::new(
  source : @source.TextIoCudaSource,
  boundary? : @precision.TextBoundary = Complete,
) -> Self {
  {
    source,
    boundary,
    buffers: [],
    functions: [],
    before: [],
    after: [],
    state: Fresh,
  }
}

///|
pub fn TextIoFrame::source_bytes(self : Self) -> Bytes {
  self.source.source_bytes()
}

///|
pub fn TextIoFrame::workspace_bytes(self : Self) -> Int64 {
  self.source.plan().boundary_workspace_bytes(self.boundary)
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
fn TextIoFrame::launch(
  self : Self,
  aot : @device.Module,
  stage : Int,
  grid : Int64,
  threads : Int,
  args : Array[@device.KernelArgument],
) -> @device.OrderedKernelLaunch raise {
  guard grid > 0L && grid <= 2147483647L else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  let function = aot.load_function(@utf8.encode(self.source.symbol(stage)))
  self.functions.push(function)
  @device.OrderedKernelLaunch::new(
    function,
    @device.KernelLaunchDimensions::new(
      grid_x=grid.to_int(),
      grid_y=1,
      grid_z=1,
      block_x=threads,
      block_y=1,
      block_z=1,
      shared_memory_bytes=0,
    ),
    @device.KernelArguments::new(args),
  )
}

///|
/// Output count and selected rows are independent of the decoder input count.
/// Input/output residuals must be distinct and borrowed through queue close.
pub fn TextIoFrame::prepare(
  self : Self,
  context : @device.Context,
  aot : @device.Module,
  counts~ : @device.Allocation,
  token_ids~ : @device.Allocation,
  output_count~ : @device.Allocation,
  selected_rows~ : @device.Allocation,
  input_residual~ : @device.Allocation,
  output_residual~ : @device.Allocation,
  embedding~ : @device.Allocation,
  norm_weight~ : @device.Allocation,
  head_weight~ : @device.Allocation,
  sampled_tokens~ : @device.Allocation,
  maximum_workspace_bytes~ : Int64,
) -> Unit raise {
  guard self.state is Fresh &&
    self.boundary is Complete &&
    self.workspace_bytes() <= maximum_workspace_bytes else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.state = Failed
  let plan = self.source.plan()
  let (_, hidden, _, rows, outputs) = plan.geometry()
  // Validate borrowed spans before allocating the private workspace.
  let counts = region(counts, 20L, 4L)
  let token_ids = region(token_ids, rows.to_int64() * 4L, 4L)
  let output_count = region(output_count, 4L, 4L)
  let selected = region(selected_rows, outputs.to_int64() * 4L, 4L)
  let input = region(input_residual, plan.residual_bytes(), 2L)
  let output = region(output_residual, plan.residual_bytes(), 2L)
  let embedding = region(embedding, plan.table_bytes(), 2L)
  let norm = region(norm_weight, hidden.to_int64() * 2L, 2L)
  let head = region(head_weight, plan.table_bytes(), 2L)
  let tokens = region(sampled_tokens, outputs.to_int64() * 4L, 4L)
  self.bind_ingress(aot, counts, token_ids, embedding, input)
  self.bind_egress(
    context, aot, counts, output_count, selected, output, norm, head, tokens,
  )
  self.state = Prepared
}

///|
pub fn TextIoFrame::prefix(
  self : Self,
) -> ArrayView[@device.OrderedKernelLaunch] raise {
  guard self.state is Prepared else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.before[:]
}

///|
pub fn TextIoFrame::suffix(
  self : Self,
) -> ArrayView[@device.OrderedKernelLaunch] raise {
  guard self.state is Prepared else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.after[:]
}

///|
pub fn TextIoFrame::logits(self : Self) -> @device.Allocation raise {
  guard self.state is Prepared && self.boundary.has_egress() else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.buffers[1]
}

///|
pub fn TextIoFrame::close(self : Self) -> Unit raise {
  if self.state is Closed {
    return
  }
  self.state = Failed
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
  self.state = Closed
}
