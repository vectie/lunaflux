///|
priv enum Phase {
  Empty
  Prepared
  Submitted
  Complete
  Failed
  Cancelled
  Closed
} derive(Eq)

///|
/// One-shot reference-audio encoding owner. Borrows input, checkpoint upload,
/// module, stream and context; owns functions, statistics, scratch and queue.
/// Preserve this owner after preparation failure so close can retry cleanup.
pub struct AudioRequest {
  priv mut phase : Phase
  priv functions : Array[@device.Function]
  priv mut source : AudioSource?
  priv mut statistics : @device.Allocation?
  priv mut scratch : @device.Allocation?
  priv mut queue : @device.OrderedKernelExecutor?
}

///|
pub fn AudioRequest::new() -> Self {
  {
    phase: Empty,
    functions: [],
    source: None,
    statistics: None,
    scratch: None,
    queue: None,
  }
}

///|
fn statistics_bytes(config : @config.MiniMaxH3AudioVaeConfig) -> Bytes {
  let result : Array[Byte] = []
  for table in [config.latent_means(), config.latent_standard_deviations()] {
    for value in table {
      let bits = value.reinterpret_as_uint()
      for byte in 0..<4 {
        result.push(((bits >> (byte * 8)) & 255U).to_byte())
      }
    }
  }
  Bytes::from_array(result)
}

///|
/// The module must be the externally AOT-compiled source.kernels.source.
/// Input is prepared stereo channel-major F32 PCM at exactly 32 kHz.
pub fn AudioRequest::prepare_resolved(
  self : Self,
  context : @device.Context,
  stream : @device.Stream,
  artifact : @device.Module,
  source : AudioSource,
  config : @config.MiniMaxH3AudioVaeConfig,
  manifest : WeightManifest,
  weights : @materialize.SegmentedDeviceWeights,
  expected_identity~ : @spec.ModelIdentity,
  input~ : @device.Allocation,
  input_offset~ : Int64,
) -> Unit raise {
  guard self.phase == Empty &&
    config.sampling_rate() == 32000 &&
    config.latent_channels() == 32 else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.phase = Failed
  self.source = Some(source)
  let weight_arguments = source.weight_arguments(
    manifest, weights, expected_identity,
  )
  let scratch = context.allocate(source.plan.graph.scratch_bytes)
  self.scratch = Some(scratch)
  let statistics = context.allocate(256L)
  self.statistics = Some(statistics)
  let bytes = statistics_bytes(config)
  statistics.copy_from_host(bytes, 0L, 0L, 256L)
  for symbol in source.kernels.symbols {
    self.functions.push(artifact.load_function(@utf8.encode(symbol)))
  }
  let launches = source.bind(
    self.functions,
    weight_arguments,
    input,
    input_offset,
    statistics,
    scratch,
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
pub fn AudioRequest::submit(self : Self) -> Unit raise @device.DeviceError {
  guard self.phase == Prepared && self.queue is Some(queue) else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorEnqueue)
  }
  self.phase = Failed
  for i in 0..<queue.kernel_count() {
    queue.enqueue(i)
  }
  queue.record_completion()
  self.phase = Submitted
}

///|
pub fn AudioRequest::poll(self : Self) -> Bool raise @device.DeviceError {
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
pub fn AudioRequest::execute(self : Self) -> Unit raise @device.DeviceError {
  self.submit()
  guard self.queue is Some(queue) else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorEnqueue)
  }
  self.phase = Failed
  queue.wait_completion()
  self.phase = Complete
}

///|
/// Planning-only borrow. The destination contains no valid data until execute
/// succeeds. Use exclusively to bind a downstream queue ordered after this one.
pub fn[T] AudioRequest::with_destination(
  self : Self,
  consume : (@device.Allocation, Int64, Int) -> T raise @device.DeviceError,
) -> T raise @device.DeviceError {
  guard (self.phase == Prepared || self.phase == Complete) &&
    self.source is Some(source) &&
    self.scratch is Some(scratch) else {
    raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
  }
  consume(
    scratch,
    source.plan.graph.offsets[source.plan.graph.output],
    source.plan.latent_time * 2,
  )
}

///|
pub impl @execution.PreparedFrame for AudioRequest with fn execute(self) {
  self.execute()
}

///|
pub impl @execution.PreparedFrame for AudioRequest with fn drain(self) {
  self.cancel()
}

///|
pub impl @execution.PreparedFrame for AudioRequest with fn release(self) {
  self.close()
}

///|
/// F32 [2*latent_time,32] normalized rows for reference conditioning. Downstream
/// consumers must finish and close their queues before this owner closes.
pub fn[T] AudioRequest::with_rows(
  self : Self,
  consume : (@device.Allocation, Int64, Int) -> T raise @device.DeviceError,
) -> T raise @device.DeviceError {
  guard self.phase == Complete &&
    self.source is Some(source) &&
    self.scratch is Some(scratch) else {
    raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
  }
  consume(
    scratch,
    source.plan.graph.offsets[source.plan.graph.output],
    source.plan.latent_time * 2,
  )
}

///|
pub fn AudioRequest::cancel(self : Self) -> Unit raise @device.DeviceError {
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
pub fn AudioRequest::close(self : Self) -> Unit raise @device.DeviceError {
  if self.phase == Closed {
    return
  }
  self.phase = Failed
  if self.queue is Some(queue) {
    queue.abort()
    queue.close()
    self.queue = None
  }
  if self.scratch is Some(scratch) {
    scratch.close()
    self.scratch = None
  }
  if self.statistics is Some(statistics) {
    statistics.close()
    self.statistics = None
  }
  while !self.functions.is_empty() {
    self.functions[self.functions.length() - 1].close()
    ignore(self.functions.pop())
  }
  self.phase = Closed
}
