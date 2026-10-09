///|
struct AudioStageLayout {
  upsample : @joint_diffusion_device.AudioUpsampleLayout
  normalized_offset : Int64
  local_bytes : Int64
  amp_bytes : Int64
}

///|
fn audio_stage_layout(
  up : @minimax_h3_block_source.AudioUpsampleSource,
  amp : @minimax_h3_block_source.AudioAmpSource,
  budget : Int64,
) -> AudioStageLayout raise @device.DeviceError {
  let u = @joint_diffusion_device.AudioUpsampleLayout::new(
    batch=up.batch,
    input_channels=up.input_channels,
    output_channels=up.output_channels,
    input_time=up.input_time,
    kernel=up.kernel,
    stride=up.stride,
    padding=up.padding,
    max_weight_bytes=budget,
  )
  guard amp.batch == up.batch &&
    amp.channels == up.output_channels &&
    amp.time == u.output_time else {
    raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
  }
  let a = @joint_diffusion_device.AudioAmpLayout::new(
    batch=amp.batch,
    channels=amp.channels,
    time=amp.time,
    max_bytes=budget,
  )
  let normalized_offset = (u.output_bytes + 255L) / 256L * 256L
  guard normalized_offset <= budget &&
    u.weight_bytes <= budget - normalized_offset else {
    raise @device.DeviceError::InvalidArgument(AllocationCreate)
  }
  let local_bytes = normalized_offset + u.weight_bytes
  guard a.scratch_bytes <= budget - local_bytes else {
    raise @device.DeviceError::InvalidArgument(AllocationCreate)
  }
  { upsample: u, normalized_offset, local_bytes, amp_bytes: a.scratch_bytes, }
}

///|
pub struct AudioDecodeStage {
  priv up : @minimax_h3_block_source.AudioUpsampleSource
  priv amp : AudioAmpProgram
  priv functions : Array[@device.Function]
  priv mut workspace : @device.Allocation?
  priv mut layout : AudioStageLayout?
  priv cache : AudioWeightCache
  priv composition : AudioDecoderLease
}

///|
pub fn AudioDecodeStage::new(
  up : @minimax_h3_block_source.AudioUpsampleSource,
  amp : @minimax_h3_block_source.AudioAmpSource,
) -> Self {
  {
    up,
    amp: AudioAmpProgram::new(amp),
    functions: [],
    workspace: None,
    layout: None,
    cache: AudioWeightCache::new(),
    composition: { held: false, },
  }
}

///|
/// Budget covers both upsample and AMP temporary storage. Retain owner on
/// partial failure and close it before retry. Modules must match source bundles.
pub fn AudioDecodeStage::prepare(
  self : Self,
  context : @device.Context,
  up_module : @device.Module,
  amp_module : @device.Module,
  max_workspace_bytes~ : Int64,
) -> Unit raise @device.DeviceError {
  guard self.functions.is_empty() && self.workspace is None else {
    raise @device.DeviceError::InvalidArgument(FunctionLoad)
  }
  let layout = audio_stage_layout(self.up, self.amp.source, max_workspace_bytes)
  acquire_functions(self.up.symbols[:], self.functions, symbol => {
    up_module.load_function(@utf8.encode(symbol))
  })
  self.workspace = Some(context.allocate(layout.local_bytes))
  self.amp.prepare(context, amp_module, max_scratch_bytes=layout.amp_bytes)
  self.layout = Some(layout)
}

///|
/// Upsample checkpoint layout is contiguous F32 g, v, bias at offset zero.
/// AMP parameters use their own admitted packed allocation. Output is external.
pub fn AudioDecodeStage::bind(
  self : Self,
  parameters : @joint_audio_amp_parameter_layout.JointAudioAmpParameterLayout,
  input~ : @device.Allocation,
  input_offset~ : Int64,
  output~ : @device.Allocation,
  output_offset~ : Int64,
  up_weights~ : @device.Allocation,
  amp_weights~ : @device.Allocation,
) -> Array[@device.OrderedKernelLaunch] raise @device.DeviceError {
  guard self.layout is Some(layout) && self.workspace is Some(workspace) else {
    raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
  }
  let u = layout.upsample
  let cache = self.cache.get(up_weights)
  let launches = u.launches(
    self.functions[0],
    self.functions[1],
    input~,
    input_offset~,
    output=workspace,
    output_offset=0L,
    normalized=cache.unwrap_or(workspace),
    normalized_offset=if cache is Some(_) {
      0L
    } else {
      layout.normalized_offset
    },
    weights=up_weights,
    g_offset=0L,
    v_offset=u.g_bytes,
    bias_offset=u.g_bytes + u.weight_bytes,
    cached=cache is Some(_),
  )
  launches.append(
    self.amp.bind(
      parameters,
      input=workspace,
      input_offset=0L,
      output~,
      output_offset~,
      weights=amp_weights,
    ),
  )
  launches
}

///|
pub fn AudioDecodeStage::close(self : Self) -> Unit raise @device.DeviceError {
  self.composition.require_idle()
  self.layout = None
  self.cache.close()
  self.amp.close()
  if self.workspace is Some(workspace) {
    workspace.close()
    self.workspace = None
  }
  release_functions(self.functions, function => function.close())
}
