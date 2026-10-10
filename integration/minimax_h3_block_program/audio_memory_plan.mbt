///|
/// Pure aggregate of all simultaneously retained decoder allocations. Raw
/// checkpoint weights and AOT module storage are external and budgeted by the
/// caller; normalization caches are additional to those raw checkpoint bytes.
pub struct AudioDecoderMemoryPlan {
  frontend_bytes : Int64
  stage_bytes : ReadOnlyArray[Int64]
  chain_bytes : Int64
  bridge_bytes : Int64
  tail_bytes : Int64
  cache_bytes : Int64
  input_bytes : Int64
  output_bytes : Int64
  total_bytes : Int64
}

///|
/// Resolve the same layouts used by prepare, before any device allocation.
/// Budgets are disjoint portions of one aggregate ceiling, not a fresh ceiling
/// for each stage. One input and one request output are included.
pub fn AudioDecoderMemoryPlan::new(
  frontend : @minimax_h3_block_source.AudioFrontendSource,
  upsample : ArrayView[@minimax_h3_block_source.AudioUpsampleSource],
  amp : ArrayView[@minimax_h3_block_source.AudioAmpSource],
  tail : @minimax_h3_block_source.AudioTailSource,
  maximum_bytes~ : Int64,
) -> Self raise @device.DeviceError {
  guard upsample.length() == 7 &&
    amp.length() == 7 &&
    frontend.batch == upsample[0].batch &&
    frontend.time == upsample[0].input_time &&
    upsample[0].input_channels == 1024 else {
    raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
  }
  let front = @joint_diffusion_device.AudioFrontendLayout::new(
    batch=frontend.batch,
    input_channels=32,
    hidden_channels=2048,
    output_channels=1024,
    time=frontend.time,
    kernel=7,
    max_bytes=maximum_bytes,
  )
  let back = @joint_diffusion_device.AudioTailLayout::new(
    batch=tail.batch,
    channels=tail.channels,
    time=tail.time,
    max_bytes=maximum_bytes,
  )
  let stage_bytes : Array[Int64] = []
  let shapes : Array[(Int64, Int64)] = []
  let caches : Array[Int64] = [front.convolution.weight_bytes]
  for i in 0..<7 {
    guard upsample[i].batch == frontend.batch &&
      (
        i == 0 ||
        (
          upsample[i].input_channels == amp[i - 1].channels &&
          upsample[i].input_time == amp[i - 1].time
        )
      ) else {
      raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
    }
    let stage = audio_stage_layout(upsample[i], amp[i], maximum_bytes)
    let layout = @joint_diffusion_device.AudioAmpLayout::new(
      batch=amp[i].batch,
      channels=amp[i].channels,
      time=amp[i].time,
      max_bytes=maximum_bytes,
    )
    stage_bytes.push(stage.local_bytes + stage.amp_bytes)
    shapes.push((stage.upsample.input_bytes, stage.upsample.output_bytes))
    caches.push(stage.upsample.weight_bytes + audio_amp_cache_offsets(layout).1)
  }
  guard tail.batch == amp[6].batch &&
    tail.channels == amp[6].channels &&
    tail.time == amp[6].time else {
    raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
  }
  caches.push(back.convolution.weight_bytes)
  let cache_bytes = audio_memory_sum(caches[:], maximum_bytes)
  let chain = audio_chain_layout(shapes[:], maximum_bytes)
  let bridge = (front.convolution.output_bytes + 255L) / 256L * 256L +
    back.activation.value_bytes
  let parts = [
    front.scratch_bytes,
    chain.bytes,
    bridge,
    back.scratch_bytes,
    cache_bytes,
    front.input_bytes,
    back.convolution.output_bytes,
  ]
  parts.append(stage_bytes[:])
  let total = audio_memory_sum(parts[:], maximum_bytes)
  {
    frontend_bytes: front.scratch_bytes,
    stage_bytes: ReadOnlyArray::from_array(stage_bytes),
    chain_bytes: chain.bytes,
    bridge_bytes: bridge,
    tail_bytes: back.scratch_bytes,
    cache_bytes,
    input_bytes: front.input_bytes,
    output_bytes: back.convolution.output_bytes,
    total_bytes: total,
  }
}

///|
fn audio_memory_sum(
  parts : ArrayView[Int64],
  maximum : Int64,
) -> Int64 raise @device.DeviceError {
  guard maximum > 0L else {
    raise @device.DeviceError::InvalidArgument(AllocationCreate)
  }
  for part in parts; remaining = maximum {
    guard part > 0L && part <= remaining else {
      raise @device.DeviceError::InvalidArgument(AllocationCreate)
    }
    continue remaining - part
  } nobreak {
    maximum - remaining
  }
}
