///|
/// Startup-only immutable normalized weights. Failed preparation retains all
/// resources for deterministic retryable cleanup; it cannot be replayed.
priv struct AudioWeightCache {
  mut allocation : @device.Allocation?
  mut queue : @device.OrderedKernelExecutor?
  mut ready : Bool
  mut source : @device.Allocation?
}

///|
fn AudioWeightCache::new() -> Self {
  { allocation: None, queue: None, ready: false, source: None, }
}

///|
fn AudioWeightCache::prepare(
  self : Self,
  context : @device.Context,
  stream : @device.Stream,
  bytes : Int64,
  budget : Int64,
  source : @device.Allocation,
  build : (@device.Allocation) -> Array[@device.OrderedKernelLaunch] raise @device.DeviceError,
) -> Unit raise @device.DeviceError {
  guard self.allocation is None &&
    self.queue is None &&
    bytes > 0L &&
    bytes <= budget else {
    raise @device.DeviceError::InvalidArgument(AllocationCreate)
  }
  let allocation = context.allocate(bytes)
  self.allocation = Some(allocation)
  self.source = Some(source)
  let launches = build(allocation)
  let queue = context.create_ordered_kernel_executor(
    stream,
    launches,
    policy=OrderedKernelEagerOnly,
  )
  self.queue = Some(queue)
  for i in 0..<queue.kernel_count() {
    queue.enqueue(i)
  }
  queue.record_completion()
  queue.wait_completion()
  queue.close()
  self.queue = None
  self.ready = true
}

///|
fn AudioWeightCache::get(
  self : Self,
  source : @device.Allocation,
) -> @device.Allocation? raise @device.DeviceError {
  if self.allocation is None {
    return None
  }
  guard self.ready &&
    self.source is Some(original) &&
    physical_equal(original, source) else {
    raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
  }
  self.allocation
}

///|
fn AudioWeightCache::close(self : Self) -> Unit raise @device.DeviceError {
  self.ready = false
  if self.queue is Some(queue) {
    queue.abort()
    queue.close()
    self.queue = None
  }
  if self.allocation is Some(allocation) {
    allocation.close()
    self.allocation = None
    self.source = None
  }
}

///|
fn audio_amp_cache_offsets(
  layout : @joint_diffusion_device.AudioAmpLayout,
) -> (ReadOnlyArray[Int64], Int64) {
  let offsets = []
  let mut end = 0L
  for convolution in layout.convolutions {
    for _ in 0..<6 {
      end = (end + 255L) / 256L * 256L
      offsets.push(end)
      end += convolution.weight_bytes
    }
  }
  (ReadOnlyArray::from_array(offsets), end)
}

///|
fn audio_amp_same_parameters(
  left : @joint_audio_amp_parameter_layout.JointAudioAmpParameterLayout,
  right : @joint_audio_amp_parameter_layout.JointAudioAmpParameterLayout,
) -> Bool {
  let l = left.regions()
  let r = right.regions()
  if l.length() != r.length() {
    return false
  }
  for i in 0..<l.length() {
    if l[i].role() != r[i].role() ||
      l[i].byte_count() != r[i].byte_count() ||
      l[i].byte_offset() != r[i].byte_offset() {
      return false
    }
  }
  true
}
