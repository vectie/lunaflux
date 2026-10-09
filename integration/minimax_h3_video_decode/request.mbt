///|
priv enum VideoRequestPhase {
  Empty
  Prepared
  Submitted
  Complete
  Cancelled
  Failed
  Closed
} derive(Eq)

///|
fn VideoRequestPhase::permits_join(self : Self) -> Bool {
  self == Prepared
}

///|
/// Owns bridge, tiled decoder, canonical pixels and execution queue. Input,
/// checkpoint weights, means/stds, context, stream and AOT module are borrowed
/// and must remain live until close. Keep this owner after any prepare failure.
pub struct VideoDecodeRequest {
  priv mut phase : VideoRequestPhase
  priv mut queue : @device.OrderedKernelExecutor?
  priv mut decoder : TiledDecoderProgram?
  priv mut bridge : @minimax_h3_block_program.VideoInputProgram?
  priv mut output : @device.Allocation?
  priv mut geometry : VideoOutput?
  priv functions : Array[@device.Function]
  priv mut input_receipt : (
    @device.Allocation,
    Int64,
    Int64,
    @joint_diffusion_loop.PreparedPlan,
  )?
}

///|
pub fn VideoDecodeRequest::new() -> Self {
  {
    phase: Empty,
    queue: None,
    decoder: None,
    bridge: None,
    output: None,
    geometry: None,
    functions: [],
    input_receipt: None,
  }
}

///|
pub fn VideoDecodeRequest::prepare(
  self : Self,
  context : @device.Context,
  stream : @device.Stream,
  bundle : VideoDecodeBundle,
  aot_module : @device.Module,
  input~ : @device.Allocation,
  input_offset~ : Int64,
  frontend_weights~ : @device.Allocation,
  blocks : ArrayView[@device.Allocation],
  head_weights~ : @device.Allocation,
  means~ : @device.Allocation,
  standard_deviations~ : @device.Allocation,
) -> Unit raise @device.DeviceError {
  guard self.phase == Empty else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.phase = Failed
  self.geometry = Some(bundle.output)
  for symbol in bundle.input.symbols {
    self.functions.push(aot_module.load_function(@utf8.encode(symbol)))
  }
  self.functions.push(
    aot_module.load_function(@utf8.encode(bundle.output.symbol)),
  )
  let bridge = @minimax_h3_block_program.VideoInputProgram::allocate(
    context,
    bundle.input,
    self.functions[0],
    self.functions[1],
    means~,
    standard_deviations~,
    max_bytes=bundle.input.value_bytes * 2L,
  )
  self.bridge = Some(bridge)
  let decoder = TiledDecoderProgram::new(bundle.decoder)
  self.decoder = Some(decoder)
  decoder.prepare(context, aot_module)
  let output = context.allocate(bundle.output.output_bytes)
  self.output = Some(output)
  let launches = bridge.bind(input, input_offset)
  bridge.with_destination((a, o) => {
    launches.append(
      decoder.bind(a, input_offset=o, frontend_weights, blocks, head_weights),
    )
  })
  decoder.with_pixels(pixels => {
    launches.push(
      @device.OrderedKernelLaunch::new(
        self.functions[2],
        dimensions(bundle.output.output_bytes),
        @device.KernelArguments::new([
          @device.KernelArgument::device_region(
            pixels,
            offset=0L,
            byte_count=bundle.output.input_bytes,
            alignment=4L,
          ),
          @device.KernelArgument::device_region(
            output,
            offset=0L,
            byte_count=bundle.output.output_bytes,
            alignment=4L,
          ),
        ]),
      ),
    )
  })
  self.queue = Some(
    context.create_ordered_kernel_executor(
      stream,
      launches,
      policy=OrderedKernelEagerOnly,
    ),
  )
  self.input_receipt = Some(
    (input, input_offset, bundle.input.value_bytes, bundle.input.plan),
  )
  self.phase = Prepared
}

///|
/// Startup join only: equal sizes are insufficient. The exact live allocation
/// owner, packed region, model, schedule and request must all match. A consumed
/// or failed decoder can never be transferred into another pipeline.
pub fn VideoDecodeRequest::matches_input(
  self : Self,
  plan : @joint_diffusion_loop.PreparedPlan,
  input : @device.Allocation,
  offset : Int64,
  byte_count : Int64,
) -> Bool {
  guard self.phase.permits_join() &&
    self.queue is Some(_) &&
    self.geometry is Some(geometry) &&
    self.input_receipt is Some((bound, bound_offset, bound_bytes, bound_plan)) else {
    return false
  }
  bound.same_storage(input) &&
  bound_offset == offset &&
  bound_bytes == byte_count &&
  bound_bytes == plan.video_elements() * 4L &&
  bound_plan.model_digest() == plan.model_digest() &&
  bound_plan.request() == plan.request() &&
  bound_plan.steps() == plan.steps() &&
  geometry.frames == plan.request().frames() &&
  geometry.height == plan.request().height() &&
  geometry.width == plan.request().width()
}

///|
/// One shot; partial submission failure is drained, never replayed.
pub fn VideoDecodeRequest::submit(
  self : Self,
) -> Unit raise @device.DeviceError {
  guard self.phase == Prepared && self.queue is Some(queue) else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorEnqueue)
  }
  self.phase = Failed
  for index in 0..<queue.kernel_count() {
    queue.enqueue(index)
  }
  queue.record_completion()
  self.phase = Submitted
}

///|
pub fn VideoDecodeRequest::poll(self : Self) -> Bool raise @device.DeviceError {
  if self.phase == Complete {
    return true
  }
  guard self.phase == Submitted && self.queue is Some(queue) else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorEnqueue)
  }
  self.phase = Failed
  let done = queue.poll_completion()
  self.phase = if done { Complete } else { Submitted }
  done
}

///|
pub fn VideoDecodeRequest::wait(self : Self) -> Unit raise @device.DeviceError {
  if self.phase == Complete {
    return
  }
  guard self.phase == Submitted && self.queue is Some(queue) else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorWait)
  }
  self.phase = Failed
  queue.wait_completion()
  self.phase = Complete
}

///|
pub fn VideoDecodeRequest::geometry(
  self : Self,
) -> VideoOutput raise @device.DeviceError {
  guard self.geometry is Some(shape) else {
    raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
  }
  shape
}

///|
/// Borrow completed canonical F32 [1,3,T,H,W]. Close any downstream executor
/// before releasing this request; callback access is not an ownership transfer.
pub fn[T] VideoDecodeRequest::with_output(
  self : Self,
  consume : (@device.Allocation) -> T raise @device.DeviceError,
) -> T raise @device.DeviceError {
  guard self.phase == Complete && self.output is Some(output) else {
    raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
  }
  consume(output)
}

///|
pub fn VideoDecodeRequest::cancel(
  self : Self,
) -> Unit raise @device.DeviceError {
  if self.phase == Closed || self.phase == Cancelled {
    return
  }
  self.phase = Failed
  if self.queue is Some(queue) {
    queue.abort()
  }
  self.phase = Cancelled
}

///|
/// Drain before freeing; failed releases retain their handles for retry.
pub fn VideoDecodeRequest::close(self : Self) -> Unit raise @device.DeviceError {
  if self.phase == Closed {
    return
  }
  self.phase = Failed
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
    decoder.close()
    self.decoder = None
  }
  if self.bridge is Some(bridge) {
    bridge.close()
    self.bridge = None
  }
  while !self.functions.is_empty() {
    self.functions[self.functions.length() - 1].close()
    let _ = self.functions.pop()
  }
  self.input_receipt = None
  self.phase = Closed
}

///|
pub fn VideoDecodeRequest::execute(
  self : Self,
) -> Unit raise @device.DeviceError {
  self.submit()
  self.wait()
}

///|
pub impl @joint_diffusion_execution.PreparedFrame for VideoDecodeRequest with fn execute(
  self,
) {
  self.execute()
}

///|
pub impl @joint_diffusion_execution.PreparedFrame for VideoDecodeRequest with fn drain(
  self,
) {
  self.cancel()
}

///|
pub impl @joint_diffusion_execution.PreparedFrame for VideoDecodeRequest with fn release(
  self,
) {
  self.close()
}
