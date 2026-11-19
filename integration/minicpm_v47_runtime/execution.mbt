///|
priv enum SessionState {
  Fresh
  Failed
  Prepared
  Closed
}

///|
/// Resident execution resources, borrowing immutable weights and the context.
/// No weight loading, HTTP, tokenizer or robot authority belongs to this owner.
pub struct Execution {
  priv program : @text.Program
  priv storage : @text.Storage
  priv frame : @text.Frame
  priv request : @request.Runner
  priv modules : ModuleSet
  priv mut stream : @device.Stream?
  priv mut state : SessionState
}

///|
pub fn Execution::new(program : @text.Program) -> Self raise {
  {
    program,
    storage: @text.Storage::new(program),
    frame: @text.Frame::new(program),
    request: @request.Runner::new(program),
    modules: ModuleSet::new(),
    stream: None,
    state: Fresh,
  }
}

///|
/// Admit the mutable payload budget before acquiring native resources. Retain
/// this owner on failure and close it before releasing context or weights.
pub fn Execution::prepare(
  self : Self,
  context : @device.Context,
  images : ModuleImages,
  layers~ : ArrayView[@decoder.LayerWeights],
  weights~ : @text.Weights,
  maximum_mutable_bytes~ : Int64,
  policy~ : @device.OrderedKernelExecutionPolicy = OrderedKernelEagerOnly,
) -> Unit raise {
  let budget = Budget::for_text(self.program, weight_bytes=0L)
  guard self.state is Fresh &&
    maximum_mutable_bytes >= budget.total_allocation_bytes else {
    raise @device.DeviceError::InvalidArgument(AllocationCreate)
  }
  self.state = Failed
  let stream = context.create_stream()
  self.stream = Some(stream)
  self.modules.prepare(context, images)
  let modules = self.modules.bindings()
  self.storage.prepare(context, maximum_device_bytes=budget.storage_bytes)
  self.frame.prepare(
    context,
    modules~,
    metadata=self.storage.metadata(),
    layers~,
    weights~,
    states=self.storage.states(),
    buffers=self.storage.buffers(),
    sampled_tokens=self.storage.sampled_tokens(),
  )
  self.request.prepare(context, stream, self.frame, self.storage, policy~)
  self.state = Prepared
}

///|
/// Borrowed request driver, retained across requests. Do not independently
/// prepare/close it; close the execution session when finished.
pub fn Execution::runner(self : Self) -> @request.Runner raise {
  guard self.state is Prepared else {
    raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
  }
  self.request
}

///|
/// Cancels active work via Runner before closing frames, buffers and modules.
/// Each successful close advances ownership, so retry preserves cleanup.
pub fn Execution::close(self : Self) -> Unit raise {
  if self.state is Closed {
    return
  }
  self.state = Failed
  self.request.close()
  self.frame.close()
  self.storage.close()
  self.modules.close()
  if self.stream is Some(stream) {
    stream.close()
    self.stream = None
  }
  self.state = Closed
}
