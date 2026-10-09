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
  priv functions : Array[@device.Function]
  priv launches : Array[@device.OrderedKernelLaunch]
  priv mut status : State
}

///|
pub fn DenseMlpFrame::new(source : @source.DenseMlpCudaSource) -> Self {
  { source, product: None, functions: [], launches: [], status: Fresh, }
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
  let (hidden, intermediate, rows) = plan.geometry()
  let product = context.allocate(plan.workspace_bytes())
  self.product = Some(product)
  let input_bytes = rows.to_int64() * hidden.to_int64() * 2L
  let arguments = [
    [
      region(counts, 20L, 4L),
      region(input, input_bytes, 2L),
      region(gate, plan.weight_bytes(), 2L),
      region(up, plan.weight_bytes(), 2L),
      region(product, plan.workspace_bytes(), 2L),
    ],
    [
      region(counts, 20L, 4L),
      region(product, plan.workspace_bytes(), 2L),
      region(down, plan.weight_bytes(), 2L),
      region(output, input_bytes, 2L),
    ],
  ]
  for stage in 0..<2 {
    let function = aot.load_function(@utf8.encode(self.source.symbol(stage)))
    self.functions.push(function)
    let width = if stage == 0 { intermediate } else { hidden }
    self.launches.push(
      @device.OrderedKernelLaunch::new(
        function,
        @device.KernelLaunchDimensions::new(
          grid_x=((rows.to_int64() * width.to_int64() + 255L) / 256L).to_int(),
          grid_y=1,
          grid_z=1,
          block_x=256,
          block_y=1,
          block_z=1,
          shared_memory_bytes=0,
        ),
        @device.KernelArguments::new(arguments[stage]),
      ),
    )
  }
  self.status = Prepared
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
  while self.functions.last() is Some(function) {
    function.close()
    ignore(self.functions.pop())
  }
  if self.product is Some(product) {
    product.close()
    self.product = None
  }
  self.status = Closed
}
