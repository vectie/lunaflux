///|
priv enum FrameState {
  Fresh
  Failed
  Prepared
  Closed
}

///|
/// Owns function handles/launch descriptions; borrows module, weights and
/// mutable buffers. Close the queue before Frame, then Storage/Loaded/module.
pub struct Frame {
  priv program : Program
  priv functions : Array[@device.Function]
  priv launches : Array[@device.OrderedKernelLaunch]
  priv mut state : FrameState
}

///|
pub fn Frame::new(program : Program) -> Self {
  { program, functions: [], launches: [], state: Fresh, }
}

///|
pub fn Frame::prepare(
  self : Self,
  context : @device.Context,
  aot~ : @device.Module,
  weights~ : Weights,
  buffers~ : Buffers,
) -> Unit raise @device.DeviceError {
  guard self.state is Fresh && buffers.rows == self.program.rows else {
    raise InvalidArgument(OrderedExecutorCreate)
  }
  for (_, region) in buffers.regions {
    context.validate_allocation_region(
      region.allocation,
      offset=region.offset,
      byte_count=region.byte_count,
      alignment=region.alignment,
    )
  }
  for name in self.program.weight_names() {
    guard weights.regions.get(name) is Some(region) else {
      raise InvalidArgument(OrderedExecutorCreate)
    }
    context.validate_allocation_region(
      region.allocation,
      offset=0L,
      byte_count=region.bytes,
      alignment=2L,
    )
    guard buffers.allocations.all(allocation => {
      !allocation.same_storage(region.allocation)
    }) else {
      raise InvalidArgument(OrderedExecutorCreate)
    }
  }
  self.state = Failed
  for entry in self.program.entries {
    self.functions.push(aot.load_function(@utf8.encode(entry.symbol)))
  }
  for stage in self.program.stages {
    let entry = self.program.entries[stage.entry]
    let arguments = stage.arguments.map(argument => {
      match argument {
        Buffer(plane) => {
          let region = buffers.region(plane)
          @device.KernelArgument::device_region(
            region.allocation,
            offset=region.offset,
            byte_count=region.byte_count,
            alignment=region.alignment,
          )
        }
        Weight(name) => {
          let region = weights.regions[name]
          @device.KernelArgument::device_region(
            region.allocation,
            offset=0L,
            byte_count=region.bytes,
            alignment=2L,
          )
        }
      }
    })
    let (grid_x, grid_y, block_x) = entry.launch
    self.launches.push(
      @device.OrderedKernelLaunch::new(
        self.functions[stage.entry],
        @device.KernelLaunchDimensions::new(
          grid_x~,
          grid_y~,
          grid_z=1,
          block_x~,
          block_y=1,
          block_z=1,
          shared_memory_bytes=0,
        ),
        @device.KernelArguments::new(arguments[:]),
      ),
    )
  }
  self.state = Prepared
}

///|
pub fn Frame::launches(
  self : Self,
) -> ArrayView[@device.OrderedKernelLaunch] raise @device.DeviceError {
  guard self.state is Prepared else {
    raise InvalidArgument(OrderedExecutorCreate)
  }
  self.launches[:]
}

///|
pub fn Frame::close(self : Self) -> Unit raise @device.DeviceError {
  if self.state is Closed {
    return
  }
  self.state = Failed
  self.launches.clear()
  while self.functions.last() is Some(function) {
    function.close()
    ignore(self.functions.pop())
  }
  self.state = Closed
}
