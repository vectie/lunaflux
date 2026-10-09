///|
/// One-shot asynchronous audio decode. Owns waveform storage and the executor;
/// borrows the decoder, weights and input. Keep the owner after any failure and
/// call close; output is unavailable until GPU completion has been observed.
pub struct AudioDecodeRequest {
  priv state : AudioRequestState
  priv mut queue : @device.OrderedKernelExecutor?
  priv mut output : @device.Allocation?
  priv mut decoder : AudioDecoder?
  priv mut input_program : AudioInputProgram?
  priv mut input_receipt : (
    @device.Allocation,
    Int64,
    Int64,
    @joint_diffusion_loop.PreparedPlan,
  )?
}

///|
pub fn AudioDecodeRequest::new() -> Self {
  {
    state: { phase: Empty, },
    queue: None,
    output: None,
    decoder: None,
    input_program: None,
    input_receipt: None,
  }
}

///|
pub fn AudioDecodeRequest::prepare(
  self : Self,
  context : @device.Context,
  stream : @device.Stream,
  decoder : AudioDecoder,
  parameters : ArrayView[
    (
      @joint_audio_amp_parameter_layout.JointAudioAmpParameterLayout,
      @device.Allocation,
      @device.Allocation,
    ),
  ],
  input~ : @device.Allocation,
  input_offset~ : Int64,
  frontend_weights~ : @device.Allocation,
  tail_weights~ : @device.Allocation,
  max_output_bytes~ : Int64,
  input_program? : AudioInputProgram,
  prepared_plan? : @joint_diffusion_loop.PreparedPlan,
) -> Unit raise @device.DeviceError {
  guard self.state.phase == Empty &&
    !decoder.closed &&
    decoder.tail.layout is Some(tail) else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  // A full pipeline consumes packed denoiser stereo rows. Raw channel-major
  // standalone decode remains supported, but deliberately has no join receipt.
  if prepared_plan is Some(plan) {
    guard input_program is Some(program) &&
      !program.closed &&
      program.layout.value_bytes == plan.audio_elements() * 4L &&
      audio_join_geometry(
        plan.request().audio_latent_frames(),
        decoder.tail.source.batch,
        decoder.tail.source.time,
        tail.convolution.output_bytes,
      ) else {
      raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
    }
  }
  decoder.require_cached_weights()
  decoder.lease.acquire()
  self.decoder = Some(decoder)
  self.state.phase = Failed
  let bytes = tail.convolution.output_bytes
  guard bytes > 0L && bytes <= max_output_bytes else {
    raise @device.DeviceError::InvalidArgument(AllocationCreate)
  }
  let output = context.allocate(bytes)
  self.output = Some(output)
  let (launches, decoder_input, decoder_input_offset) = match input_program {
    Some(program) => {
      guard decoder.frontend.layout is Some(front) &&
        program.layout.value_bytes == front.input_bytes else {
        raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
      }
      program.lease.acquire()
      self.input_program = Some(program)
      (
        program.bind(input, input_offset),
        program.storage,
        program.layout.value_bytes,
      )
    }
    None => ([], input, input_offset)
  }
  launches.append(
    decoder.bind(
      parameters,
      input=decoder_input,
      input_offset=decoder_input_offset,
      output~,
      output_offset=0L,
      frontend_weights~,
      tail_weights~,
    ),
  )
  self.queue = Some(
    context.create_ordered_kernel_executor(
      stream,
      launches,
      policy=OrderedKernelEagerOnly,
    ),
  )
  self.state.phase = Prepared
  if prepared_plan is Some(plan) {
    self.input_receipt = Some(
      (input, input_offset, plan.audio_elements() * 4L, plan),
    )
  }
}

///|
/// Full joint requests require a packed-input bridge and this exact prepared
/// storage binding. This is startup metadata comparison, never a token-step scan.
pub fn AudioDecodeRequest::matches_input(
  self : Self,
  plan : @joint_diffusion_loop.PreparedPlan,
  input : @device.Allocation,
  offset : Int64,
  byte_count : Int64,
) -> Bool {
  guard self.state.permits_join() &&
    self.queue is Some(_) &&
    self.input_program is Some(program) &&
    !program.closed &&
    self.input_receipt is Some((bound, bound_offset, bound_bytes, bound_plan)) else {
    return false
  }
  bound.same_storage(input) &&
  bound_offset == offset &&
  bound_bytes == byte_count &&
  program.layout.value_bytes == bound_bytes &&
  bound_bytes == plan.audio_elements() * 4L &&
  bound_plan.model_digest() == plan.model_digest() &&
  bound_plan.request() == plan.request() &&
  bound_plan.steps() == plan.steps()
}

///|
fn AudioRequestState::permits_join(self : Self) -> Bool {
  self.phase == Prepared
}

///|
/// Enqueue once without a host wait between kernels. Partial enqueue failure is
/// terminal for this request and is drained by cancel/close, never replayed.
pub fn AudioDecodeRequest::submit(
  self : Self,
) -> Unit raise @device.DeviceError {
  guard self.queue is Some(queue) else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorEnqueue)
  }
  self.state.submit(() => {
    for index in 0..<queue.kernel_count() {
      queue.enqueue(index)
    }
    queue.record_completion()
  })
}

///|
pub fn AudioDecodeRequest::poll(self : Self) -> Bool raise @device.DeviceError {
  guard self.queue is Some(queue) else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorEnqueue)
  }
  self.state.poll(() => queue.poll_completion())
}

///|
/// Borrows completed channel-major F32 waveform storage. Do not retain a
/// downstream queue beyond this owner's lifetime; close downstream queues first.
pub fn[T] AudioDecodeRequest::with_output(
  self : Self,
  consume : (@device.Allocation) -> T raise @device.DeviceError,
) -> T raise @device.DeviceError {
  guard self.state.phase == Complete && self.output is Some(output) else {
    raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
  }
  consume(output)
}

///|
/// Drains submitted work before reporting cancellation. Does not free buffers.
pub fn AudioDecodeRequest::cancel(
  self : Self,
) -> Unit raise @device.DeviceError {
  self.state.cancel(() => if self.queue is Some(queue) { queue.abort() })
}

///|
/// Retryable release: queue leases are closed before owned output. Borrowed
/// decoder and stage owners remain the caller's responsibility.
pub fn AudioDecodeRequest::close(self : Self) -> Unit raise @device.DeviceError {
  if self.state.phase == Closed {
    return
  }
  self.state.phase = Failed
  if self.queue is Some(queue) {
    queue.abort()
    queue.close()
    self.queue = None
  }
  if self.output is Some(output) {
    output.close()
    self.output = None
  }
  if self.decoder is Some(decoder) {
    decoder.lease.held = false
    self.decoder = None
  }
  if self.input_program is Some(program) {
    program.lease.held = false
    self.input_program = None
  }
  self.state.phase = Closed
  self.input_receipt = None
}
