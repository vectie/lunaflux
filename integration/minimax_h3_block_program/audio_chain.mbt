///|
struct AudioChainLayout {
  stride : Int64
  bytes : Int64
}

///|
fn audio_chain_layout(
  shapes : ArrayView[(Int64, Int64)],
  budget : Int64,
) -> AudioChainLayout raise @device.DeviceError {
  guard shapes.length() == 7 else {
    raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
  }
  let largest = for i in 0..<7; largest = 0L {
    let (input, output) = shapes[i]
    guard input > 0L && output > 0L && (i == 0 || shapes[i - 1].1 == input) else {
      raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
    }
    continue if i < 6 { Int64::max(largest, output) } else { largest }
  } nobreak {
    largest
  }
  guard largest <= 9223372036854775552L else {
    raise @device.DeviceError::InvalidArgument(AllocationCreate)
  }
  let stride = (largest + 255L) / 256L * 256L
  guard budget > 0L && stride <= budget / 2L else {
    raise @device.DeviceError::InvalidArgument(AllocationCreate)
  }
  { stride, bytes: stride * 2L, }
}

///|
/// Borrows seven prepared decode stages. Owns only two reusable inter-stage
/// buffers; final output is caller-owned. Close queue leases before this owner,
/// then close stages. This chain starts after AudioVAE's conv_pre.
pub struct AudioDecodeChain {
  priv stages : ReadOnlyArray[AudioDecodeStage]
  priv scratch : @device.Allocation
  priv layout : AudioChainLayout
  priv mut closed : Bool
  priv composition : AudioDecoderLease
  priv children : ReadOnlyArray[AudioDecoderLease]
  priv mut children_claimed : Bool
}

///|
pub fn AudioDecodeChain::allocate(
  context : @device.Context,
  stages : ArrayView[AudioDecodeStage],
  max_intermediate_bytes~ : Int64,
) -> Self raise @device.DeviceError {
  let shapes = []
  for i, stage in stages {
    guard stage.layout is Some(layout) else {
      raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
    }
    if i > 0 {
      let previous = stages[i - 1]
      guard previous.amp.source.channels == stage.up.input_channels &&
        previous.amp.source.time == stage.up.input_time &&
        previous.up.batch == stage.up.batch else {
        raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
      }
    }
    shapes.push((layout.upsample.input_bytes, layout.upsample.output_bytes))
  }
  let layout = audio_chain_layout(shapes, max_intermediate_bytes)
  let children = stages.map(stage => stage.composition)
  audio_composition_preflight(children)
  let scratch = context.allocate(layout.bytes)
  audio_composition_claim(children)
  {
    stages: ReadOnlyArray::from_array(stages.to_owned()),
    scratch,
    layout,
    closed: false,
    composition: { held: false, },
    children: ReadOnlyArray::from_array(children),
    children_claimed: true,
  }
}

///|
/// Compose all seven 102-launch stages. Parameter tuples are (AMP layout,
/// upsampler g/v/bias allocation, AMP allocation), all packed at offset zero.
pub fn AudioDecodeChain::bind(
  self : Self,
  parameters : ArrayView[
    (
      @joint_audio_amp_parameter_layout.JointAudioAmpParameterLayout,
      @device.Allocation,
      @device.Allocation,
    ),
  ],
  input~ : @device.Allocation,
  input_offset~ : Int64,
  output~ : @device.Allocation,
  output_offset~ : Int64,
) -> Array[@device.OrderedKernelLaunch] raise @device.DeviceError {
  guard !self.closed && parameters.length() == self.stages.length() else {
    raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
  }
  let launches : Array[@device.OrderedKernelLaunch] = []
  for i, stage in self.stages {
    let (source, source_offset) = if i == 0 {
      (input, input_offset)
    } else {
      (self.scratch, ((i - 1) % 2).to_int64() * self.layout.stride)
    }
    let (destination, destination_offset) = if i == self.stages.length() - 1 {
      (output, output_offset)
    } else {
      (self.scratch, (i % 2).to_int64() * self.layout.stride)
    }
    let (amp, up_weights, amp_weights) = parameters[i]
    launches.append(
      stage.bind(
        amp,
        input=source,
        input_offset=source_offset,
        output=destination,
        output_offset=destination_offset,
        up_weights~,
        amp_weights~,
      ),
    )
  }
  launches
}

///|
pub fn AudioDecodeChain::close(self : Self) -> Unit raise @device.DeviceError {
  self.composition.require_idle()
  self.closed = true
  self.scratch.close()
  if self.children_claimed {
    audio_composition_release(self.children[:])
    self.children_claimed = false
  }
}
