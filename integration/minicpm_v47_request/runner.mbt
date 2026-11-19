///|
priv enum RunnerState {
  Fresh
  Prepared
  Failed
  Closed
}

///|
/// Owns two prepared queues and host metadata. The context, stream, frame and
/// storage are borrowed and must outlive this owner. Use the same Program for
/// all three owners. Close Runner before Frame/Storage/stream/context.
pub struct Runner {
  priv host : Host
  priv executors : Array[@device.OrderedKernelExecutor]
  priv completed : FixedArray[Bool]
  priv mut bindings : ReadOnlyArray[(Port, @device.Allocation)]
  priv mut context : @device.Context?
  priv mut active : Int32
  priv mut ready : Bool
  priv mut state : RunnerState
}

///|
pub fn Runner::new(program : @text.Program) -> Self raise RequestError {
  {
    host: Host::new(Profile::from_program(program)),
    executors: [],
    completed: FixedArray::make(2, false),
    bindings: [],
    context: None,
    active: -1,
    ready: false,
    state: Fresh,
  }
}

///|
/// Prepare outside request execution. Static page/slot metadata is uploaded
/// once. Partial failure retains acquired queues until close is called.
pub fn Runner::prepare(
  self : Self,
  context : @device.Context,
  stream : @device.Stream,
  frame : @text.Frame,
  storage : @text.Storage,
  policy~ : @device.OrderedKernelExecutionPolicy = OrderedKernelEagerOnly,
) -> Unit raise {
  self.prepare_launches(
    context,
    stream,
    storage,
    prefill=frame.launches(Prefill),
    decode=frame.launches(Decode),
    policy~,
  )
}

///|
/// Prepare explicit model-owned queues, retaining the same host metadata and
/// step law. Optional input transformations belong in the caller's launch
/// program, not callbacks or model branches inside the request controller.
pub fn Runner::prepare_launches(
  self : Self,
  context : @device.Context,
  stream : @device.Stream,
  storage : @text.Storage,
  prefill~ : ArrayView[@device.OrderedKernelLaunch],
  decode~ : ArrayView[@device.OrderedKernelLaunch],
  policy~ : @device.OrderedKernelExecutionPolicy = OrderedKernelEagerOnly,
) -> Unit raise {
  guard self.state is Fresh else { raise RequestError::InvalidLifecycle }
  self.state = Failed
  self.context = Some(context)
  let metadata = storage.metadata()
  let decoder = metadata.decoder
  let attention = decoder.attention
  self.bindings = [
    (Counts, attention.counts),
    (TokenIds, metadata.token_ids),
    (CanvasPositions, attention.canvas_positions),
    (CausalPositions, attention.causal_positions),
    (RowOffsets, attention.row_offsets),
    (SequenceLength, attention.sequence_lengths),
    (PageOffsets, attention.page_offsets),
    (PhysicalPages, attention.physical_pages),
    (RequestSlots, decoder.request_slots),
    (Reset, decoder.reset),
    (OutputCount, metadata.output_count),
    (SelectedRows, metadata.selected_rows),
    (SampledToken, storage.sampled_tokens()),
  ]
  for index in 0..<self.bindings.length() {
    let (port, allocation) = self.bindings[index]
    context.validate_allocation_region(
      allocation,
      offset=0L,
      byte_count=self.host.packet(port).length().to_int64(),
      alignment=4L,
    )
  }
  self.executors.push(
    context.create_ordered_kernel_executor(stream, prefill, policy~),
  )
  self.executors.push(
    context.create_ordered_kernel_executor(stream, decode, policy~),
  )
  for index in 0..<self.bindings.length() {
    let (port, allocation) = self.bindings[index]
    if port is (PhysicalPages | RequestSlots) {
      let bytes = self.host.packet(port)
      context.copy_from_fixed_host(
        allocation,
        bytes,
        0L,
        0L,
        bytes.length().to_int64(),
      )
    }
  }
  self.ready = true
  self.state = Prepared
}

///|
pub fn Runner::begin(
  self : Self,
  prompt : @prompt.PreparedTextPrompt,
  maximum_new_tokens~ : Int32,
  stop_token_ids~ : ReadOnlyArray[Int32],
  context_ceiling? : Int32,
) -> Unit raise RequestError {
  guard self.state is Prepared && self.active == -1 else {
    raise InvalidLifecycle
  }
  self.host.begin(
    prompt,
    maximum_new_tokens~,
    stop_token_ids~,
    context_ceiling?,
  )
}

///|
/// Stage and enqueue one chunk/token. False means the request is finished.
/// Only metadata uploads occur here; no weights or queue descriptions change.
pub fn Runner::submit_next(self : Self) -> Bool raise {
  guard self.state is Prepared && self.active == -1 else {
    raise RequestError::InvalidLifecycle
  }
  if !self.host.stage_next() {
    return false
  }
  self.state = Failed
  let context = self.context.unwrap()
  for index in 0..<self.bindings.length() {
    let (port, allocation) = self.bindings[index]
    if !(port is (PhysicalPages | RequestSlots | SampledToken)) {
      context.copy_from_fixed_host(
        allocation,
        self.host.packet(port),
        0L,
        0L,
        self.host.live_bytes(port).to_int64(),
      )
    }
  }
  let index = if self.host.staged_phase() is Prefill { 0 } else { 1 }
  let executor = self.executors[index]
  if self.completed[index] {
    executor.reset()
    self.completed[index] = false
  }
  self.host.mark_submitted()
  // Publish cleanup authority before the first potentially partial enqueue.
  self.active = index
  match executor.execution_mode() {
    OrderedKernelEager =>
      for kernel in 0..<executor.kernel_count() {
        executor.enqueue(kernel)
      }
    OrderedKernelCaptured => executor.launch_captured()
  }
  executor.record_completion()
  self.state = Prepared
  true
}

///|
fn Runner::retire(self : Self) -> Unit raise {
  self.state = Failed
  let mut token = 0
  if self.host.expects_token() {
    for index in 0..<self.bindings.length() {
      let (port, allocation) = self.bindings[index]
      if port is SampledToken {
        let packet = self.host.packet(SampledToken)
        self.context.unwrap().copy_to_fixed_host(allocation, packet, 0L, 0L, 4L)
        token = read_i32(packet)
      }
    }
  }
  self.host.complete_step(token)
  self.completed[self.active] = true
  self.active = -1
  self.state = Prepared
}

///|
/// Nonblocking event query. On true the chunk has committed, including its
/// optional sampled token; the caller can observe output or submit next work.
pub fn Runner::poll(self : Self) -> Bool raise {
  guard self.state is Prepared && self.active >= 0 else {
    raise RequestError::InvalidLifecycle
  }
  if !self.executors[self.active].poll_completion() {
    return false
  }
  self.retire()
  true
}

///|
pub fn Runner::wait(self : Self) -> Unit raise {
  guard self.state is Prepared && self.active >= 0 else {
    raise RequestError::InvalidLifecycle
  }
  self.executors[self.active].wait_completion()
  self.retire()
}

///|
/// Retire any submitted device work before releasing request state. Also
/// recovers a failed submission/readback after its cleanup barrier succeeds.
pub fn Runner::cancel(self : Self) -> Unit raise {
  guard self.state is (Prepared | Failed) && self.ready else {
    raise RequestError::InvalidLifecycle
  }
  if self.active >= 0 {
    self.executors[self.active].abort()
    self.completed[self.active] = true
    self.active = -1
  }
  self.host.stop_quiescent(Cancelled)
  self.state = Prepared
}

///|
pub fn Runner::is_finished(self : Self) -> Bool {
  self.host.is_finished()
}

///|
pub fn Runner::finish_reason(self : Self) -> @inference.FinishReason? {
  self.host.finish_reason()
}

///|
pub fn Runner::sampled_tokens(self : Self) -> Int32 {
  self.host.sampled_tokens()
}

///|
pub fn Runner::cached_tokens(self : Self) -> Int32 {
  self.host.cached_tokens()
}

///|
pub fn Runner::emitted_tokens(self : Self) -> ArrayView[Int32] {
  self.host.emitted_tokens()
}

///|
/// Retryable deterministic cleanup. Borrowed resources are not closed here.
pub fn Runner::close(self : Self) -> Unit raise {
  if self.state is Closed {
    return
  }
  self.state = Failed
  if self.active >= 0 {
    self.executors[self.active].abort()
    self.active = -1
  }
  while self.executors.last() is Some(executor) {
    executor.close()
    ignore(self.executors.pop())
  }
  self.bindings = []
  self.context = None
  self.ready = false
  self.state = Closed
}
