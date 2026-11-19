///|
pub(all) struct Metadata {
  counts : @device.Allocation
  row_offsets : @device.Allocation
  request_slots : @device.Allocation
  reset : @device.Allocation
}

///|
pub(all) struct Weights {
  qkv : @device.Allocation
  gate : @device.Allocation
  decay : @device.Allocation
  beta : @device.Allocation
  output : @device.Allocation
  convolution : @device.Allocation
  norm : @device.Allocation
  a_log : @device.Allocation
  dt_bias : @device.Allocation
}

///|
/// Borrowed views into persistent history and FP32 matrix state arenas.
pub struct States {
  priv convolution : @device.Allocation
  priv recurrent : @device.Allocation
  priv convolution_offset : Int64
  priv recurrent_offset : Int64
}

///|
pub fn States::new(
  convolution : @device.Allocation,
  recurrent : @device.Allocation,
  convolution_offset~ : Int64 = 0L,
  recurrent_offset~ : Int64 = 0L,
) -> Self {
  { convolution, recurrent, convolution_offset, recurrent_offset, }
}

///|
priv enum FrameState {
  Fresh
  Failed
  Prepared
  Closed
}

///|
/// Owns nine functions and prepared launch descriptions; every module/buffer is
/// borrowed. Close enclosing executors before the frame, then borrowed owners.
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
/// Prepare once. Caller supplies valid counts/CSR, exclusive request slots and
/// reset flags shared by convolution and recurrence. Close partial failures.
pub fn Frame::prepare(
  self : Self,
  context : @device.Context,
  aot : @device.Module,
  metadata~ : Metadata,
  weights~ : Weights,
  states~ : States,
  input~ : @device.Allocation,
  output~ : @device.Allocation,
  workspace~ : @device.Allocation,
) -> Unit raise @device.DeviceError {
  guard self.status is Fresh else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  context.validate_allocation_region(
    workspace,
    offset=0L,
    byte_count=self.program.workspace_bytes(),
    alignment=16L,
  )
  let conv_bytes = self.program.region(ConvolutionState).byte_count
  let recurrent_bytes = self.program.region(RecurrentState).byte_count
  context.validate_allocation_region(
    states.convolution,
    offset=states.convolution_offset,
    byte_count=conv_bytes,
    alignment=2L,
  )
  context.validate_allocation_region(
    states.recurrent,
    offset=states.recurrent_offset,
    byte_count=recurrent_bytes,
    alignment=4L,
  )
  guard !states.convolution.same_storage(states.recurrent) ||
    states.convolution_offset >= states.recurrent_offset + recurrent_bytes ||
    states.recurrent_offset >= states.convolution_offset + conv_bytes else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  let borrowed = [
    metadata.counts,
    metadata.row_offsets,
    metadata.request_slots,
    metadata.reset,
    weights.qkv,
    weights.gate,
    weights.decay,
    weights.beta,
    weights.output,
    weights.convolution,
    weights.norm,
    weights.a_log,
    weights.dt_bias,
    states.convolution,
    states.recurrent,
    input,
    output,
  ]
  guard borrowed.all(buffer => !buffer.same_storage(workspace)) else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.status = Failed
  let allocation = fn(slot) {
    match slot {
      Counts => metadata.counts
      RowOffsets => metadata.row_offsets
      RequestSlots => metadata.request_slots
      Reset => metadata.reset
      Input => input
      Output => output
      QkvWeight => weights.qkv
      GateWeight => weights.gate
      DecayWeight => weights.decay
      BetaWeight => weights.beta
      OutputWeight => weights.output
      ConvolutionWeight => weights.convolution
      NormWeight => weights.norm
      ALog => weights.a_log
      DtBias => weights.dt_bias
      ConvolutionState => states.convolution
      RecurrentState => states.recurrent
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
        let offset = match slot {
          ConvolutionState => states.convolution_offset
          RecurrentState => states.recurrent_offset
          _ => region.workspace_offset.unwrap_or(0L)
        }
        @device.KernelArgument::device_region(
          allocation(slot),
          offset~,
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
  self.status = Prepared
}

///|
pub fn Frame::launches(
  self : Self,
) -> ArrayView[@device.OrderedKernelLaunch] raise @device.DeviceError {
  guard self.status is Prepared else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
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
