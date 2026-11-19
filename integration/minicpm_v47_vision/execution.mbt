///|
priv enum ExecutionState {
  Fresh
  Failed
  Prepared
  Closed
}

///|
/// Borrowed BF16 [rows,2048] output. Valid until next submit/cancel/close;
/// consumers must finish device use before reusing this single-flight owner.
pub struct FeatureView {
  allocation : @device.Allocation
  rows : Int32
}

///|
pub fn FeatureView::allocation(self : Self) -> @device.Allocation {
  self.allocation
}

///|
pub fn FeatureView::rows(self : Self) -> Int32 {
  self.rows
}

///|
pub struct Execution {
  priv program : Program
  priv storage : Storage
  priv frame : Frame
  priv mut aot : @device.Module?
  priv mut stream : @device.Stream?
  priv mut executor : @device.OrderedKernelExecutor?
  priv mut state : ExecutionState
  priv mut ready : Bool
  priv mut active : Bool
  priv mut completed : Bool
  priv mut pending_rows : Int32
  priv mut feature_rows : Int32
}

///|
pub fn Execution::new(program : Program) -> Self {
  {
    program,
    storage: Storage::new(program),
    frame: Frame::new(program),
    aot: None,
    stream: None,
    executor: None,
    state: Fresh,
    ready: false,
    active: false,
    completed: false,
    pending_rows: 0,
    feature_rows: 0,
  }
}

///|
/// Borrows context/weights; owns module, stream, mutable storage, frame and
/// queue. Admission occurs before any native resource creation. Close after
/// partial failure, even when prepare did not return successfully.
pub fn Execution::prepare(
  self : Self,
  context : @device.Context,
  cubin : Bytes,
  weights : Weights,
  maximum_mutable_bytes~ : Int64,
  policy~ : @device.OrderedKernelExecutionPolicy = OrderedKernelEagerOnly,
) -> Unit raise {
  guard self.state is Fresh &&
    maximum_mutable_bytes >= self.program.storage_bytes() else {
    raise @device.DeviceError::InvalidArgument(AllocationCreate)
  }
  guard !cubin.is_empty() else {
    raise @device.DeviceError::InvalidArgument(ModuleLoad)
  }
  self.state = Failed
  let stream = context.create_stream()
  self.stream = Some(stream)
  let aot = context.load_module(cubin)
  self.aot = Some(aot)
  self.storage.prepare(context, maximum_device_bytes=maximum_mutable_bytes)
  self.frame.prepare(context, aot~, weights~, buffers=self.storage.buffers())
  self.executor = Some(
    context.create_ordered_kernel_executor(
      stream,
      self.frame.launches(),
      policy~,
    ),
  )
  self.ready = true
  self.state = Prepared
}

///|
pub fn Execution::is_ready(self : Self) -> Bool {
  self.state is Prepared && !self.active
}

///|
pub fn Execution::submit(self : Self, input : Input) -> Unit raise {
  guard self.is_ready() &&
    input.capacity == self.program.rows &&
    input.crop_capacity == self.program.crop_rows else {
    raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
  }
  self.state = Failed
  self.feature_rows = 0
  let buffers = self.storage.buffers()
  for i in 0..<input.uploads.length() {
    let upload = input.uploads[i]
    if upload.bytes.is_empty() {
      continue
    }
    let region = buffers.region(upload.plane)
    region.allocation.copy_from_host(
      upload.bytes,
      0L,
      region.offset,
      upload.bytes.length().to_int64(),
    )
  }
  let executor = self.executor.unwrap()
  if self.completed {
    executor.reset()
    self.completed = false
  }
  self.pending_rows = input.outputs
  // Retain cleanup authority before the first potentially partial enqueue.
  self.active = true
  match executor.execution_mode() {
    OrderedKernelEager =>
      for i in 0..<executor.kernel_count() {
        executor.enqueue(i)
      }
    OrderedKernelCaptured => executor.launch_captured()
  }
  executor.record_completion()
  self.state = Prepared
}

///|
fn Execution::retire(self : Self) -> Unit {
  self.active = false
  self.completed = true
  self.feature_rows = self.pending_rows
  self.state = Prepared
}

///|
pub fn Execution::poll(self : Self) -> Bool raise {
  guard self.state is Prepared && self.active else {
    raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
  }
  if !self.executor.unwrap().poll_completion() {
    return false
  }
  self.retire()
  true
}

///|
pub fn Execution::wait(self : Self) -> Unit raise {
  guard self.state is Prepared && self.active else {
    raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
  }
  self.executor.unwrap().wait_completion()
  self.retire()
}

///|
pub fn Execution::features(
  self : Self,
) -> FeatureView raise @device.DeviceError {
  guard self.is_ready() && self.feature_rows > 0 else {
    raise InvalidArgument(AotKernelLaunch)
  }
  {
    allocation: self.storage.buffers().region(Features).allocation,
    rows: self.feature_rows,
  }
}

///|
/// Startup-only borrowed storage for downstream binding. Contents are valid
/// only through features() after completion, until this owner is reused.
pub fn Execution::feature_buffer(
  self : Self,
) -> @device.Allocation raise @device.DeviceError {
  guard self.ready && self.state is (Prepared | Failed) else {
    raise InvalidArgument(AotKernelLaunch)
  }
  self.storage.buffers().region(Features).allocation
}

///|
/// A failed enqueue is reusable only after the executor's cleanup barrier.
pub fn Execution::cancel(self : Self) -> Unit raise {
  guard self.ready && self.state is (Prepared | Failed) else {
    raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
  }
  if self.active {
    self.executor.unwrap().abort()
    self.completed = true
    self.active = false
  }
  self.feature_rows = 0
  self.pending_rows = 0
  self.state = Prepared
}

///|
pub fn Execution::close(self : Self) -> Unit raise {
  if self.state is Closed {
    return
  }
  self.state = Failed
  if self.executor is Some(executor) {
    if self.active {
      executor.abort()
      self.active = false
    }
    executor.close()
    self.executor = None
  }
  self.frame.close()
  self.storage.close()
  if self.aot is Some(aot) {
    aot.close()
    self.aot = None
  }
  if self.stream is Some(stream) {
    stream.close()
    self.stream = None
  }
  self.ready = false
  self.feature_rows = 0
  self.state = Closed
}
