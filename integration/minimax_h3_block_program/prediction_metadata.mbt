///|
struct PredictionMetadataLayout {
  times : ReadOnlyArray[Int64]
  patterns : ReadOnlyArray[(Int64, Int64)]
  counts : ReadOnlyArray[Int]
  bytes : Int64
}

///|
/// Pure startup layout: each equality pattern is stored once, while distinct
/// timestep values remain per step. Never reuse step zero's modulation indices
/// for a later step whose target/reference times have collapsed together.
fn prediction_metadata_layout(
  schedule : @minimax_h3_sequence.TimestepSchedule,
  budget : Int64,
) -> PredictionMetadataLayout raise @device.DeviceError {
  let times : Array[Int64] = []
  let patterns : Array[(Int64, Int64)] = []
  let counts = Array::make(schedule.patterns().length(), 0)
  let mut bytes = 0L
  fn reserve(size : Int64) -> Int64 raise @device.DeviceError {
    guard size > 0L && bytes <= budget && size <= budget - bytes else {
      raise @device.DeviceError::InvalidArgument(AllocationCreate)
    }
    let offset = bytes
    bytes += size
    offset
  }
  for step in schedule.steps() {
    let n = step.values.length()
    guard n > 0 && n <= 4 && step.pattern >= 0 && step.pattern < counts.length() else {
      raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
    }
    guard counts[step.pattern] == 0 || counts[step.pattern] == n else {
      raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
    }
    counts[step.pattern] = n
    times.push(reserve(n.to_int64() * 4L))
  }
  for index, pattern in schedule.patterns() {
    guard counts[index] > 0 &&
      pattern.indices.length() == pattern.combined_adaln.length() else {
      raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
    }
    let size = pattern.indices.length().to_int64() * 4L
    let block_offset = reserve(size)
    let final_offset = reserve(size)
    patterns.push((block_offset, final_offset))
  }
  {
    times: ReadOnlyArray::from_array(times),
    patterns: ReadOnlyArray::from_array(patterns),
    counts: ReadOnlyArray::from_array(counts),
    bytes,
  }
}

///|
fn prediction_time_bytes(values : ArrayView[Float]) -> Bytes {
  let bytes : Array[Byte] = Array::make(values.length() * 4, 0)
  for i, value in values {
    let bits = value.reinterpret_as_uint()
    for b in 0..<4 {
      bytes[i * 4 + b] = (bits >> (b * 8)).to_byte()
    }
  }
  Bytes::from_array(bytes)
}

///|
/// Request-local immutable timestep/index upload. Keep this owner after failed
/// preparation and close it after consuming queues; there are no per-step copies.
pub struct PredictionMetadata {
  priv plan : @joint_diffusion_loop.PreparedPlan
  priv sequence : @minimax_h3_sequence.Sequence
  priv schedule : @minimax_h3_sequence.TimestepSchedule
  priv layout : PredictionMetadataLayout
  priv mut allocation : @device.Allocation?
  priv blocks : Array[@joint_diffusion_device.ModulationBindings]
  priv finals : Array[@joint_diffusion_device.ModulationBindings]
  priv mut ready : Bool
  priv mut started : Bool
}

///|
pub fn PredictionMetadata::new(
  plan : @joint_diffusion_loop.PreparedPlan,
  sequence : @minimax_h3_sequence.Sequence,
  visual_noise~ : Double,
  audio_noise~ : Double,
  vision_text_rows~ : ArrayView[Int],
  max_metadata_bytes~ : Int64,
) -> Self raise {
  let steps = plan.steps()
  let schedule = sequence.schedule(
    video=steps[:].map(step => step.video_timestep),
    audio=steps[:].map(step => step.audio_timestep),
    visual_noise~,
    audio_noise~,
    vision_text_rows~,
    max_steps=steps.length(),
    max_index_bytes=max_metadata_bytes,
  )
  let layout = prediction_metadata_layout(schedule, max_metadata_bytes)
  {
    plan,
    sequence,
    schedule,
    layout,
    allocation: None,
    blocks: [],
    finals: [],
    ready: false,
    started: false,
  }
}

///|
pub fn PredictionMetadata::distinct_timestep_counts(
  self : Self,
) -> ReadOnlyArray[Int] {
  let counts : Array[Int] = []
  for count in self.layout.counts {
    if !counts.contains(count) {
      counts.push(count)
    }
  }
  counts.sort()
  ReadOnlyArray::from_array(counts)
}

///|
pub fn PredictionMetadata::prepare(
  self : Self,
  context : @device.Context,
) -> Unit raise {
  guard !self.started else {
    raise @device.DeviceError::InvalidArgument(AllocationCreate)
  }
  self.started = true
  let allocation = context.allocate(self.layout.bytes)
  self.allocation = Some(allocation)
  for index, step in self.schedule.steps() {
    let bytes = prediction_time_bytes(step.values[:])
    allocation.copy_from_host(
      bytes,
      0L,
      self.layout.times[index],
      bytes.length().to_int64(),
    )
  }
  for index, pattern in self.schedule.patterns() {
    let offsets = self.layout.patterns[index]
    let count = self.layout.counts[index]
    self.blocks.push(
      @joint_diffusion_device.ModulationBindings::upload(
        context,
        allocation,
        pattern.combined_adaln[:],
        offset=offsets.0,
        width=self.sequence.assembly().width(),
        parameter_rows=count * 3,
      ),
    )
    self.finals.push(
      @joint_diffusion_device.ModulationBindings::upload(
        context,
        allocation,
        pattern.indices[:],
        offset=offsets.1,
        width=self.sequence.assembly().width(),
        parameter_rows=count,
      ),
    )
  }
  self.ready = true
}

///|
pub fn PredictionMetadata::close(self : Self) -> Unit raise @device.DeviceError {
  self.ready = false
  self.started = true
  if self.allocation is Some(allocation) {
    allocation.close()
    self.allocation = None
  }
  self.blocks.clear()
  self.finals.clear()
}
