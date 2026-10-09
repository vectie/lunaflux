///|
/// Decode channel-major, destandardized F32 latents to stereo waveform.
/// Borrows prepared stages; owns only frontend/body bridge storage.
pub struct AudioDecoder {
  priv frontend : AudioFrontendProgram
  priv chain : AudioDecodeChain
  priv tail : AudioTailProgram
  priv bridge : @device.Allocation
  priv tail_offset : Int64
  priv mut closed : Bool
  priv lease : AudioDecoderLease
  priv children : ReadOnlyArray[AudioDecoderLease]
  priv mut children_claimed : Bool
}

///|
pub fn AudioDecoder::allocate(
  context : @device.Context,
  frontend : AudioFrontendProgram,
  chain : AudioDecodeChain,
  tail : AudioTailProgram,
  max_bridge_bytes~ : Int64,
) -> Self raise @device.DeviceError {
  guard !chain.closed &&
    frontend.layout is Some(front) &&
    tail.layout is Some(back) &&
    chain.stages.length() == 7 else {
    raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
  }
  let first = chain.stages[0]
  let last = chain.stages[6]
  guard frontend.source.batch == first.up.batch &&
    frontend.source.time == first.up.input_time &&
    first.up.input_channels == 1024 &&
    tail.source.batch == last.amp.source.batch &&
    tail.source.channels == last.amp.source.channels &&
    tail.source.time == last.amp.source.time else {
    raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
  }
  let tail_offset = (front.convolution.output_bytes + 255L) / 256L * 256L
  guard max_bridge_bytes >= tail_offset &&
    back.activation.value_bytes <= max_bridge_bytes - tail_offset else {
    raise @device.DeviceError::InvalidArgument(AllocationCreate)
  }
  let children = [frontend.composition, chain.composition, tail.composition]
  audio_composition_preflight(children)
  let bridge = context.allocate(tail_offset + back.activation.value_bytes)
  audio_composition_claim(children)
  {
    frontend,
    chain,
    tail,
    bridge,
    tail_offset,
    closed: false,
    lease: { held: false, },
    children: ReadOnlyArray::from_array(children),
    children_claimed: true,
  }
}

///|
/// 3 frontend + 714 body + 5 tail launches. Preparation only: caller owns the
/// ordered executor and must drain/close it before this owner or borrowed stages.
pub fn AudioDecoder::bind(
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
  frontend_weights~ : @device.Allocation,
  tail_weights~ : @device.Allocation,
) -> Array[@device.OrderedKernelLaunch] raise @device.DeviceError {
  guard !self.closed else {
    raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
  }
  let launches = self.frontend.bind(
    input~,
    input_offset~,
    output=self.bridge,
    output_offset=0L,
    weights=frontend_weights,
  )
  launches.append(
    self.chain.bind(
      parameters,
      input=self.bridge,
      input_offset=0L,
      output=self.bridge,
      output_offset=self.tail_offset,
    ),
  )
  launches.append(
    self.tail.bind(
      input=self.bridge,
      input_offset=self.tail_offset,
      output~,
      output_offset~,
      weights=tail_weights,
    ),
  )
  launches
}

///|
pub fn AudioDecoder::close(self : Self) -> Unit raise @device.DeviceError {
  self.lease.require_idle()
  self.closed = true
  self.bridge.close()
  if self.children_claimed {
    audio_composition_release(self.children[:])
    self.children_claimed = false
  }
}
