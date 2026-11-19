///|
pub(all) struct Weights {
  router : @device.Allocation
  routed_bank : @device.Allocation
  shared_gate : @device.Allocation
  shared_up : @device.Allocation
  shared_down : @device.Allocation
  shared_gate_projection : @device.Allocation
}

///|
priv enum FrameState {
  Fresh
  Failed
  Prepared
  Closed
}

///|
/// Owns nine AOT function handles and their prepared launches. All allocations,
/// module and context are borrowed. Close the enclosing executor before Frame;
/// then close workspace/weights/module/context through their existing owners.
pub struct Frame {
  priv program : Program
  priv functions : Array[@device.Function]
  priv launches : Array[@device.OrderedKernelLaunch]
  priv mut status : FrameState
}

///|
pub fn Frame::new(program : Program) -> Self {
  { program, functions: [], launches: [], status: Fresh, }
}

///|
/// Prepare once; reuse the workspace serially across layers. No file operations,
/// allocations or argument reconstruction are needed when executing the queue.
/// On partial failure, close the frame before releasing its borrowed resources.
pub fn Frame::prepare(
  self : Self,
  context : @device.Context,
  aot : @device.Module,
  counts~ : @device.Allocation,
  input~ : @device.Allocation,
  output~ : @device.Allocation,
  weights~ : Weights,
  workspace~ : @device.Allocation,
) -> Unit raise @device.DeviceError {
  guard self.status is Fresh else {
    raise InvalidArgument(OrderedExecutorCreate)
  }
  context.validate_allocation_region(
    workspace,
    offset=0L,
    byte_count=self.program.workspace_bytes(),
    alignment=16L,
  )
  let borrowed = [
    counts,
    input,
    output,
    weights.router,
    weights.routed_bank,
    weights.shared_gate,
    weights.shared_up,
    weights.shared_down,
    weights.shared_gate_projection,
  ]
  guard borrowed.all(buffer => !buffer.same_storage(workspace)) else {
    raise InvalidArgument(OrderedExecutorCreate)
  }
  self.status = Failed
  let allocation = fn(slot) {
    match slot {
      Counts => counts
      Input => input
      Output => output
      RouterWeight => weights.router
      RoutedBank => weights.routed_bank
      SharedGateWeight => weights.shared_gate
      SharedUpWeight => weights.shared_up
      SharedDownWeight => weights.shared_down
      SharedGateProjectionWeight => weights.shared_gate_projection
      _ => workspace
    }
  }
  for stage in self.program.stages() {
    let function = aot.load_function(@utf8.encode(stage.symbol()))
    self.functions.push(function)
    let arguments = stage
      .arguments()
      .map(slot => {
        let region = self.program.region(slot)
        @device.KernelArgument::device_region(
          allocation(slot),
          offset=region.workspace_offset.unwrap_or(0L),
          byte_count=region.byte_count,
          alignment=region.alignment,
        )
      })
    let (grid_x, grid_y, block_x) = stage.launch()
    self.launches.push(
      @device.OrderedKernelLaunch::new(
        function,
        @device.KernelLaunchDimensions::new(
          grid_x~,
          grid_y~,
          grid_z=1,
          block_x~,
          block_y=1,
          block_z=1,
          shared_memory_bytes=0,
        ),
        @device.KernelArguments::new(arguments),
      ),
    )
  }
  let expert_map = self.program.operators.routed_experts().expert_map()
  let bytes = FixedArray::makei(expert_map.length() * 4, index => {
    (expert_map[index / 4].reinterpret_as_uint() >> (index % 4 * 8)).to_byte()
  })
  context.copy_from_fixed_host(
    workspace,
    bytes,
    0L,
    self.program.region(ExpertMap).workspace_offset.unwrap(),
    bytes.length().to_int64(),
  )
  self.status = Prepared
}

///|
pub fn Frame::launches(
  self : Self,
) -> ArrayView[@device.OrderedKernelLaunch] raise @device.DeviceError {
  guard self.status is Prepared else {
    raise InvalidArgument(OrderedExecutorCreate)
  }
  self.launches[:]
}

///|
pub fn Frame::close(self : Self) -> Unit raise @device.DeviceError {
  if self.status is Closed {
    return
  }
  self.status = Failed
  self.launches.clear()
  while self.functions.last() is Some(function) {
    function.close()
    ignore(self.functions.pop())
  }
  self.status = Closed
}
