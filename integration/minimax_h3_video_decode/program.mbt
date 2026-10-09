///|
/// Holds all successful acquisitions after prepare failure. Caller owns the
/// admitted module and input/weights. Close consuming queues before this owner.
pub struct FrontendProgram {
  priv frontend : Frontend
  priv functions : Array[@device.Function]
  priv mut intermediate : @device.Allocation?
  priv mut output : @device.Allocation?
  priv mut ready : Bool
}

///|
pub fn FrontendProgram::new(frontend : Frontend) -> Self {
  { frontend, functions: [], intermediate: None, output: None, ready: false, }
}

///|
/// AOT module must have been admitted against this exact source offline.
pub fn FrontendProgram::prepare(
  self : Self,
  context : @device.Context,
  aot_module : @device.Module,
) -> Unit raise @device.DeviceError {
  guard self.functions.is_empty() &&
    self.intermediate is None &&
    self.output is None else {
    raise @device.DeviceError::InvalidArgument(FunctionLoad)
  }
  for symbol in self.frontend.symbols {
    self.functions.push(aot_module.load_function(@utf8.encode(symbol)))
  }
  self.intermediate = Some(context.allocate(self.frontend.input_bytes))
  self.output = Some(context.allocate(self.frontend.output_bytes))
  self.ready = true
}

///|
pub fn FrontendProgram::bind(
  self : Self,
  input : @device.Allocation,
  input_offset~ : Int64,
  weights : @device.Allocation,
) -> Array[@device.OrderedKernelLaunch] raise @device.DeviceError {
  guard self.ready &&
    self.intermediate is Some(intermediate) &&
    self.output is Some(output) &&
    self.functions.length() == 2 else {
    raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
  }
  let regions = self.frontend.weights.copies
  fn weight(i : Int) -> @device.KernelArgument raise @device.DeviceError {
    @device.KernelArgument::device_region(
      weights,
      offset=regions[i].destination_offset,
      byte_count=regions[i].byte_count,
      alignment=4L,
    )
  }
  [
    @device.OrderedKernelLaunch::new(
      self.functions[0],
      dimensions(self.frontend.input_bytes),
      @device.KernelArguments::new([
        @device.KernelArgument::device_region(
          input,
          offset=input_offset,
          byte_count=self.frontend.input_bytes,
          alignment=4L,
        ),
        weight(0),
        weight(1),
        @device.KernelArgument::device_region(
          intermediate,
          offset=0L,
          byte_count=self.frontend.input_bytes,
          alignment=4L,
        ),
      ]),
    ),
    @device.OrderedKernelLaunch::new(
      self.functions[1],
      dimensions(self.frontend.output_bytes),
      @device.KernelArguments::new([
        @device.KernelArgument::device_region(
          intermediate,
          offset=0L,
          byte_count=self.frontend.input_bytes,
          alignment=4L,
        ),
        weight(2),
        weight(3),
        weight(4),
        @device.KernelArgument::device_region(
          output,
          offset=0L,
          byte_count=self.frontend.output_bytes,
          alignment=4L,
        ),
      ]),
    ),
  ]
}

///|
/// Borrow only for downstream queue binding. The caller must order execution
/// after the two frontend launches; this does not assert completion/readiness.
pub fn[T] FrontendProgram::with_output(
  self : Self,
  action : (@device.Allocation) -> T raise @device.DeviceError,
) -> T raise @device.DeviceError {
  guard self.ready && self.output is Some(output) else {
    raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
  }
  action(output)
}

///|
pub fn FrontendProgram::close(self : Self) -> Unit raise @device.DeviceError {
  self.ready = false
  if self.output is Some(output) {
    output.close()
    self.output = None
  }
  if self.intermediate is Some(intermediate) {
    intermediate.close()
    self.intermediate = None
  }
  while !self.functions.is_empty() {
    self.functions[self.functions.length() - 1].close()
    let _ = self.functions.pop()
  }
}

///|
fn dimensions(
  bytes : Int64,
) -> @device.KernelLaunchDimensions raise @device.DeviceError {
  @device.KernelLaunchDimensions::new(
    grid_x=((bytes / 4L + 255L) / 256L).to_int(),
    grid_y=1,
    grid_z=1,
    block_x=256,
    block_y=1,
    block_z=1,
    shared_memory_bytes=0,
  )
}
