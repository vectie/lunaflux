///|
priv enum State {
  Fresh
  Prepared
  Failed
  Closed
}

///|
/// Owns a materialized shared SwiGLU product and two AOT functions. A caller
/// borrows its launch view into a larger queue; close that queue first.
pub struct DenseMlpFrame {
  priv source : @source.DenseMlpCudaSource
  priv mut product : @device.Allocation?
  priv mut partials : @device.Allocation?
  // Borrowed output for BF16 reduction; unlike partials, not owned here.
  priv mut tensor_output : @device.Allocation?
  priv functions : Array[@device.Function]
  priv launches : Array[@device.OrderedKernelLaunch]
  priv mut status : State
}

///|
pub fn DenseMlpFrame::new(source : @source.DenseMlpCudaSource) -> Self {
  {
    source,
    product: None,
    partials: None,
    tensor_output: None,
    functions: [],
    launches: [],
    status: Fresh,
  }
}

///|
pub fn DenseMlpFrame::source_bytes(self : Self) -> Bytes {
  self.source.source_bytes()
}

///|
pub fn DenseMlpFrame::workspace_bytes(self : Self) -> Int64 {
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
pub fn DenseMlpFrame::prepare(
  self : Self,
  context : @device.Context,
  aot : @device.Module,
  counts~ : @device.Allocation,
  input~ : @device.Allocation,
  gate~ : @device.Allocation,
  up~ : @device.Allocation,
  down~ : @device.Allocation,
  output~ : @device.Allocation,
  maximum_workspace_bytes~ : Int64,
) -> Unit raise {
  guard self.status is Fresh &&
    maximum_workspace_bytes >= self.workspace_bytes() else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.status = Failed
  let plan = self.source.plan()
  let (hidden, _, rows) = plan.geometry()
  let product = context.allocate(plan.product_bytes())
  self.product = Some(product)
  let input_bytes = rows.to_int64() * hidden.to_int64() * 2L
  let partial = plan.publication() == ReduceFloat32BeforeRound
  let down_output = if partial {
    let buffer = context.allocate(input_bytes * 2L)
    self.partials = Some(buffer)
    buffer
  } else {
    output
  }
  let arguments = [
    [
      region(counts, 20L, 4L),
      region(input, input_bytes, 2L),
      region(gate, plan.weight_bytes(), 2L),
      region(up, plan.weight_bytes(), 2L),
      region(product, plan.product_bytes(), 2L),
    ],
    [
      region(counts, 20L, 4L),
      region(product, plan.product_bytes(), 2L),
      region(down, plan.weight_bytes(), 2L),
      region(
        down_output,
        if partial {
          input_bytes * 2L
        } else {
          input_bytes
        },
        if partial {
          4L
        } else {
          2L
        },
      ),
    ],
  ]
  for stage in self.source.execution_order() {
    let function = aot.load_function(@utf8.encode(self.source.symbol(stage)))
    self.functions.push(function)
    let (grid, threads) = self.source.launch(stage)
    self.launches.push(
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
        @device.KernelArguments::new(arguments[stage]),
      ),
    )
  }
  if partial {
    let function = aot.load_function(@utf8.encode(self.source.symbol(2)))
    self.functions.push(function)
    self.launches.push(
      @device.OrderedKernelLaunch::new(
        function,
        @device.KernelLaunchDimensions::new(
          grid_x=((input_bytes / 2L + 255L) / 256L).to_int(),
          grid_y=1,
          grid_z=1,
          block_x=256,
          block_y=1,
          block_z=1,
          shared_memory_bytes=0,
        ),
        @device.KernelArguments::new([
          region(counts, 20L, 4L),
          region(down_output, input_bytes * 2L, 4L),
          region(output, input_bytes, 2L),
        ]),
      ),
    )
  }
  if plan.publication() == ReduceBFloat16AfterRound {
    self.tensor_output = Some(output)
  }
  self.status = Prepared
}

///|
pub fn DenseMlpFrame::tensor_width(self : Self) -> Int? {
  if self.source.plan().publication() != LocalBFloat16 {
    let (hidden, _, _) = self.source.plan().geometry()
    Some(hidden)
  } else {
    None
  }
}

///|
pub fn DenseMlpFrame::tensor_is_bfloat16(self : Self) -> Bool {
  self.source.plan().publication() == ReduceBFloat16AfterRound
}

///|
pub fn DenseMlpFrame::tensor_boundary(
  self : Self,
) -> (Int, @device.Allocation, Int)? raise @device.DeviceError {
  guard self.status is Prepared else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  if self.tensor_output is Some(buffer) && self.tensor_width() is Some(width) {
    Some((self.launches.length(), buffer, width))
  } else if self.partials is Some(buffer) && self.tensor_width() is Some(width) {
    Some((self.launches.length() - 1, buffer, width))
  } else {
    None
  }
}

///|
pub fn DenseMlpFrame::launches(
  self : Self,
) -> ArrayView[@device.OrderedKernelLaunch] raise @device.DeviceError {
  guard self.status is Prepared else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.launches[:]
}

///|
pub fn DenseMlpFrame::close(self : Self) -> Unit raise @device.DeviceError {
  if self.status is Closed {
    return
  }
  self.status = Failed
  self.launches.clear()
  self.tensor_output = None
  while self.functions.last() is Some(function) {
    function.close()
    ignore(self.functions.pop())
  }
  if self.product is Some(product) {
    product.close()
    self.product = None
  }
  if self.partials is Some(partials) {
    partials.close()
    self.partials = None
  }
  self.status = Closed
}
