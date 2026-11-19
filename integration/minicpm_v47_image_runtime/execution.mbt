///|
priv enum SessionState {
  Fresh
  Prepared
  Failed
  Closed
}

///|
priv enum Work {
  Idle
  Vision
  Text
}

///|
pub suberror ExecutionError {
  InvalidLifecycle
  InvalidRequest
}

///|
/// Single-flight image/text execution. Owns all mutable resources and queues;
/// borrows the external context and immutable text/vision weights.
pub struct Execution {
  priv budget : Budget
  priv storage : @text.Storage
  priv text_frame : @text.Frame
  priv bridge : @multimodal.Frame
  priv vision : @vision.Execution
  priv request : @request.Runner
  priv modules : @runtime.ModuleSet
  priv mut scatter_module : @device.Module?
  priv mut stream : @device.Stream?
  priv mut context : @device.Context?
  priv mut pending : @multimodal.ImagePrompt?
  priv mut work : Work
  priv mut ready : Bool
  priv mut state : SessionState
}

///|
pub fn Execution::new(
  program : @multimodal.Program,
  vision : @vision.Program,
) -> Self raise {
  let text = program.text_program()
  {
    budget: Budget::new(
      program,
      vision,
      text_weight_bytes=0L,
      vision_weight_bytes=0L,
    ),
    storage: @text.Storage::new(text),
    text_frame: @text.Frame::new(text),
    bridge: @multimodal.Frame::new(program),
    vision: @vision.Execution::new(vision),
    request: @request.Runner::new(text),
    modules: @runtime.ModuleSet::new(),
    scatter_module: None,
    stream: None,
    context: None,
    pending: None,
    work: Idle,
    ready: false,
    state: Fresh,
  }
}

///|
/// Admit the full mutable payload before allocating. On partial failure, retain
/// and close this owner before releasing borrowed weights or context.
pub fn Execution::prepare(
  self : Self,
  context : @device.Context,
  images : ModuleImages,
  layers~ : ArrayView[@decoder.LayerWeights],
  text_weights~ : @text.Weights,
  vision_weights~ : @vision.Weights,
  maximum_mutable_bytes~ : Int64,
  policy~ : @device.OrderedKernelExecutionPolicy = OrderedKernelEagerOnly,
) -> Unit raise {
  guard self.state is Fresh &&
    maximum_mutable_bytes >= self.budget.mutable_bytes else {
    raise @device.DeviceError::InvalidArgument(AllocationCreate)
  }
  guard !images.vision.is_empty() && !images.image_scatter.is_empty() else {
    raise @device.DeviceError::InvalidArgument(ModuleLoad)
  }
  self.state = Failed
  self.context = Some(context)
  let stream = context.create_stream()
  self.stream = Some(stream)
  self.modules.prepare(context, images.text)
  self.storage.prepare(
    context,
    maximum_device_bytes=self.budget.text.storage_bytes,
  )
  self.text_frame.prepare(
    context,
    modules=self.modules.bindings(),
    metadata=self.storage.metadata(),
    layers~,
    weights=text_weights,
    states=self.storage.states(),
    buffers=self.storage.buffers(),
    sampled_tokens=self.storage.sampled_tokens(),
  )
  self.vision.prepare(
    context,
    images.vision,
    vision_weights,
    maximum_mutable_bytes=self.budget.vision_storage_bytes,
    policy~,
  )
  let scatter_module = context.load_module(images.image_scatter)
  self.scatter_module = Some(scatter_module)
  self.bridge.prepare(
    context,
    aot=scatter_module,
    text_frame=self.text_frame,
    text_storage=self.storage,
    features=self.vision.feature_buffer(),
    maximum_metadata_bytes=self.budget.image_metadata_bytes,
  )
  self.request.prepare_launches(
    context,
    stream,
    self.storage,
    prefill=self.bridge.launches(Prefill),
    decode=self.bridge.launches(Decode),
    policy~,
  )
  self.ready = true
  self.state = Prepared
}

///|
fn Execution::require_idle(self : Self) -> Unit raise ExecutionError {
  guard self.state is Prepared &&
    (self.work is Idle || (self.work is Text && self.request.is_finished())) else {
    raise ExecutionError::InvalidLifecycle
  }
}

///|
/// Starts vision asynchronously. Poll/wait once for its completion before
/// submit_next; features remain device-resident through the decoder request.
pub fn Execution::begin_image(
  self : Self,
  prompt : @multimodal.ImagePrompt,
  input : @vision.Input,
  maximum_new_tokens~ : Int32,
  stop_token_ids~ : ReadOnlyArray[Int32],
  context_ceiling? : Int32,
) -> Unit raise {
  self.require_idle()
  guard prompt.feature_rows() == input.feature_rows() else {
    raise ExecutionError::InvalidRequest
  }
  self.request.begin(
    prompt.prompt(),
    maximum_new_tokens~,
    stop_token_ids~,
    context_ceiling?,
  )
  self.state = Failed
  self.pending = Some(prompt)
  self.work = Vision
  self.vision.submit(input)
  self.state = Prepared
}

///|
pub fn Execution::begin_text(
  self : Self,
  prompt : @prompt.PreparedTextPrompt,
  maximum_new_tokens~ : Int32,
  stop_token_ids~ : ReadOnlyArray[Int32],
  context_ceiling? : Int32,
) -> Unit raise {
  self.require_idle()
  self.state = Failed
  self.bridge.clear_prompt(self.context.unwrap())
  self.request.begin(
    prompt,
    maximum_new_tokens~,
    stop_token_ids~,
    context_ceiling?,
  )
  self.work = Text
  self.state = Prepared
}

///|
pub fn Execution::submit_next(self : Self) -> Bool raise {
  guard self.state is Prepared && self.work is Text else {
    raise ExecutionError::InvalidLifecycle
  }
  self.state = Failed
  let submitted = self.request.submit_next()
  self.state = Prepared
  submitted
}

///|
fn Execution::bind_features(self : Self) -> Unit raise {
  self.bridge.bind_prompt(
    self.context.unwrap(),
    self.pending.unwrap(),
    self.vision.features(),
  )
  self.pending = None
  self.work = Text
}

///|
/// Nonblocking completion for the pending vision operation or decoder step.
pub fn Execution::poll(self : Self) -> Bool raise {
  guard self.state is Prepared && !(self.work is Idle) else {
    raise ExecutionError::InvalidLifecycle
  }
  self.state = Failed
  let complete = match self.work {
    Vision => {
      let complete = self.vision.poll()
      if complete {
        self.bind_features()
      }
      complete
    }
    Text => self.request.poll()
    Idle => false
  }
  self.state = Prepared
  complete
}

///|
pub fn Execution::wait(self : Self) -> Unit raise {
  guard self.state is Prepared && !(self.work is Idle) else {
    raise ExecutionError::InvalidLifecycle
  }
  self.state = Failed
  match self.work {
    Vision => {
      self.vision.wait()
      self.bind_features()
    }
    Text => self.request.wait()
    Idle => ()
  }
  self.state = Prepared
}

///|
/// Retire both queues before allowing image storage or request metadata reuse.
pub fn Execution::cancel(self : Self) -> Unit raise {
  guard self.ready && self.state is (Prepared | Failed) else {
    raise ExecutionError::InvalidLifecycle
  }
  self.state = Failed
  self.request.cancel()
  self.vision.cancel()
  self.pending = None
  self.work = Idle
  self.state = Prepared
}

///|
pub fn Execution::is_finished(self : Self) -> Bool {
  self.state is Prepared &&
  (self.work is Idle || (self.work is Text && self.request.is_finished()))
}

///|
pub fn Execution::finish_reason(self : Self) -> @inference.FinishReason? {
  self.request.finish_reason()
}

///|
pub fn Execution::emitted_tokens(self : Self) -> ArrayView[Int32] {
  self.request.emitted_tokens()
}

///|
pub fn Execution::cached_tokens(self : Self) -> Int32 {
  self.request.cached_tokens()
}

///|
pub fn Execution::sampled_tokens(self : Self) -> Int32 {
  self.request.sampled_tokens()
}

///|
pub fn Execution::close(self : Self) -> Unit raise {
  if self.state is Closed {
    return
  }
  self.state = Failed
  self.ready = false
  self.request.close()
  self.bridge.close()
  self.text_frame.close()
  self.storage.close()
  self.vision.close()
  if self.scatter_module is Some(aot) {
    aot.close()
    self.scatter_module = None
  }
  self.modules.close()
  if self.stream is Some(stream) {
    stream.close()
    self.stream = None
  }
  self.context = None
  self.pending = None
  self.work = Idle
  self.state = Closed
}
