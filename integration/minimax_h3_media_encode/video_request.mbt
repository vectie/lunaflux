///|
/// Complete spatial-tile and temporal-clip video VAE encoding owner. Input,
/// standard-normal noise and checkpoint upload are borrowed until release.
/// No implicit seed or claim of PyTorch RNG parity is made.
pub struct VideoRequest {
  priv mut phase : Phase
  priv functions : Array[@device.Function]
  priv mut source : VideoSource?
  priv mut statistics : @device.Allocation?
  priv mut scratch : @device.Allocation?
  priv mut queue : @device.OrderedKernelExecutor?
}

///|
pub fn VideoRequest::new() -> Self {
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
fn video_statistics_bytes(config : @config.MiniMaxH3VideoVaeConfig) -> Bytes {
  let bytes : Array[Byte] = []
  for table in [config.latent_means(), config.latent_standard_deviations()] {
    for value in table {
      let bits = value.reinterpret_as_uint()
      for i in 0..<4 {
        bytes.push(((bits >> (i * 8)) & 255U).to_byte())
      }
    }
  }
  Bytes::from_array(bytes)
}

///|
/// Uses the completed generic segmented upload. The admitted AOT module must
/// be source.kernels.source. Input is ImageNet F32 [3,T,H,W]; noise is F32
/// [24,latent_time,H/16,W/16], sampled externally before request execution.
pub fn VideoRequest::prepare_resolved(
  self : Self,
  context : @device.Context,
  stream : @device.Stream,
  artifact : @device.Module,
  source : VideoSource,
  config : @config.MiniMaxH3VideoVaeConfig,
  manifest : WeightManifest,
  weights : @materialize.SegmentedDeviceWeights,
  expected_identity~ : @spec.ModelIdentity,
  input~ : @device.Allocation,
  input_offset~ : Int64,
  noise~ : @device.Allocation,
  noise_offset~ : Int64,
) -> Unit raise {
  guard self.phase == Empty &&
    config.latent_channels() == 24 &&
    config.clip_length() == 17 &&
    config.token_drop() == 3 else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.phase = Failed
  self.source = Some(source)
  let resolved = resolve_weight_arguments(
    source.plan.graph,
    VideoVae,
    manifest,
    weights,
    expected_identity,
  )
  let scratch = context.allocate(source.plan.graph.scratch_bytes)
  self.scratch = Some(scratch)
  let statistics = context.allocate(192L)
  self.statistics = Some(statistics)
  statistics.copy_from_host(video_statistics_bytes(config), 0L, 0L, 192L)
  for name in source.kernels.symbols {
    self.functions.push(artifact.load_function(@utf8.encode(name)))
  }
  let launches = source.bind(
    self.functions,
    resolved,
    input,
    input_offset,
    noise,
    noise_offset,
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
pub fn VideoRequest::submit(self : Self) -> Unit raise @device.DeviceError {
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
pub fn VideoRequest::poll(self : Self) -> Bool raise @device.DeviceError {
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
pub fn VideoRequest::execute(self : Self) -> Unit raise @device.DeviceError {
  self.submit()
  guard self.queue is Some(queue) else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorEnqueue)
  }
  self.phase = Failed
  queue.wait_completion()
  self.phase = Complete
}

///|
/// Planning-only destination [latentT,H/32,W/32,96]. Downstream launch order
/// must enforce producer completion. No initialized-output authority is given.
pub fn[T] VideoRequest::with_destination(
  self : Self,
  consume : (@device.Allocation, Int64, Int) -> T raise @device.DeviceError,
) -> T raise @device.DeviceError {
  guard (self.phase == Prepared || self.phase == Complete) &&
    self.source is Some(s) &&
    self.scratch is Some(scratch) else {
    raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
  }
  consume(
    scratch,
    s.plan.graph.offsets[s.plan.graph.output],
    (s.plan.graph.values[s.plan.graph.output].elements / 96L).to_int(),
  )
}

///|
pub fn[T] VideoRequest::with_rows(
  self : Self,
  consume : (@device.Allocation, Int64, Int) -> T raise @device.DeviceError,
) -> T raise @device.DeviceError {
  guard self.phase == Complete else {
    raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
  }
  self.with_destination(consume)
}

///|
pub fn VideoRequest::cancel(self : Self) -> Unit raise @device.DeviceError {
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
pub fn VideoRequest::close(self : Self) -> Unit raise @device.DeviceError {
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

///|
pub impl @execution.PreparedFrame for VideoRequest with fn execute(self) {
  self.execute()
}

///|
pub impl @execution.PreparedFrame for VideoRequest with fn drain(self) {
  self.cancel()
}

///|
pub impl @execution.PreparedFrame for VideoRequest with fn release(self) {
  self.close()
}
