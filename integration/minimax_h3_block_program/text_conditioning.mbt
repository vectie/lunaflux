///|
priv enum ConditioningPhase {
  Empty
  Prepared
  Complete
  Failed
} derive(Eq)

///|
/// Request-static text conditioning owner. The encoder rows are borrowed;
/// projection/refinement output is a fresh allocation and cannot alias them.
/// Keep this owner after preparation/execution failure and explicitly close it.
pub struct TextConditioningProgram {
  priv source : @minimax_h3_block_source.RefinerSource
  priv refiner : RefinerProgram
  priv mut output : @device.Allocation?
  priv mut queue : @device.OrderedKernelExecutor?
  priv mut phase : ConditioningPhase
}

///|
pub fn TextConditioningProgram::new(
  source : @minimax_h3_block_source.RefinerSource,
) -> Self {
  {
    source,
    refiner: RefinerProgram::new(source),
    output: None,
    queue: None,
    phase: Empty,
  }
}

///|
fn text_conditioning_bytes(
  rows : Int,
  width : Int,
  budget : Int64,
) -> Int64 raise @device.DeviceError {
  guard rows > 0 &&
    width > 0 &&
    budget > 0L &&
    rows.to_int64() <= budget / 2L / width.to_int64() else {
    raise @device.DeviceError::InvalidArgument(AllocationCreate)
  }
  rows.to_int64() * width.to_int64() * 2L
}

///|
/// Build one eager preparation queue; capture is unnecessary for work executed
/// once per request. Context weights must be packed at allocation offset zero.
pub fn TextConditioningProgram::prepare(
  self : Self,
  context : @device.Context,
  stream : @device.Stream,
  aot_module : @device.Module,
  layers : ArrayView[
    (@minimax_h3_weights.RefinerWeightPacking, @device.Allocation),
  ],
  encoded~ : @device.Allocation,
  encoded_offset~ : Int64,
  weights~ : @device.Allocation,
  max_workspace_bytes~ : Int64,
  max_output_bytes~ : Int64,
) -> Unit raise @device.DeviceError {
  guard self.phase == Empty else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.phase = Failed
  let bytes = text_conditioning_bytes(
    self.source.rows,
    self.source.model.denoiser().hidden_size(),
    max_output_bytes,
  )
  self.refiner.prepare(context, aot_module, max_workspace_bytes~)
  let output = context.allocate(bytes)
  self.output = Some(output)
  let launches = self.refiner.bind_context(
    layers,
    encoded~,
    encoded_offset~,
    weights~,
    state=output,
    state_offset=0L,
  )
  self.queue = Some(
    context.create_ordered_kernel_executor(
      stream,
      launches,
      policy=OrderedKernelEagerOnly,
    ),
  )
  self.phase = Prepared
}

///|
fn execute_conditioning_once(
  phase : ConditioningPhase,
  run : () -> Unit raise @device.DeviceError,
) -> ConditioningPhase raise @device.DeviceError {
  guard phase == Prepared else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorEnqueue)
  }
  run()
  Complete
}

///|
/// Finish all preparation before exposing conditioning to denoising. Failed
/// or completed preparation cannot accidentally run twice on updated state.
pub fn TextConditioningProgram::execute(
  self : Self,
) -> Unit raise @device.DeviceError {
  guard self.phase == Prepared && self.queue is Some(queue) else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorEnqueue)
  }
  let previous = self.phase
  self.phase = Failed
  self.phase = execute_conditioning_once(previous, () => {
    for index in 0..<queue.kernel_count() {
      queue.enqueue(index)
    }
    queue.record_completion()
    queue.wait_completion()
  })
}

///|
/// Connect completed text-only conditioning to per-step latent assembly.
/// Requests with reference-media rows require a separate complete conditioning
/// assembly and are rejected rather than silently filling those rows with text.
pub fn TextConditioningProgram::bind_input(
  self : Self,
  input : InputProgram,
  packing : @minimax_h3_weights.InputWeightPacking,
  weights~ : @device.Allocation,
) -> Array[@device.OrderedKernelLaunch] raise @device.DeviceError {
  guard self.phase == Complete && self.output is Some(output) else {
    raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
  }
  guard input.source.model == self.source.model &&
    input.source.sequence.segments()[0].count == self.source.rows &&
    input.source.sequence.assembly().input_rows(Conditioning) ==
    self.source.rows else {
    raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
  }
  input.bind(
    packing,
    weights~,
    conditioning=Some(output),
    conditioning_offset=0L,
  )
}

///|
/// Close downstream denoising queues first. Abort/drain this preparation queue
/// before releasing functions or buffers; failed closes retain retryable owners.
pub fn TextConditioningProgram::close(
  self : Self,
) -> Unit raise @device.DeviceError {
  self.phase = Failed
  if self.queue is Some(queue) {
    queue.abort()
    queue.close()
    self.queue = None
  }
  self.refiner.close()
  if self.output is Some(output) {
    output.close()
    self.output = None
  }
  self.phase = Empty
}
