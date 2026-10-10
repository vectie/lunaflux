///|
/// Checkpoint-backed complete AudioVAE decoder bootstrap. Modules and runtime
/// scratch are owned here; immutable raw weights are borrowed under the caller's
/// WeightLease. Close every prepared request before this owner, then the lease.
pub struct CheckpointAudioDecoder {
  priv sources : @source.AudioDecoderSources
  priv memory : @program.AudioDecoderMemoryPlan
  priv frontend : @program.AudioFrontendProgram
  priv tail : @program.AudioTailProgram
  priv stages : Array[@program.AudioDecodeStage]
  priv modules : Array[@device.Module]
  priv mut chain : @program.AudioDecodeChain?
  priv mut decoder : @program.AudioDecoder?
  priv parameters : Array[
    (
      @parameters.JointAudioAmpParameterLayout,
      @device.Allocation,
      @device.Allocation,
    ),
  ]
  priv mut front_weights : @device.Allocation?
  priv mut tail_weights : @device.Allocation?
  priv mut started : Bool
  priv mut ready : Bool
}

///|
pub fn CheckpointAudioDecoder::new(
  sources : @source.AudioDecoderSources,
  maximum_workspace_bytes~ : Int64,
) -> Self raise {
  let memory = @program.AudioDecoderMemoryPlan::new(
    sources.frontend,
    sources.upsample[:],
    sources.amp[:],
    sources.tail,
    maximum_bytes=maximum_workspace_bytes,
  )
  {
    sources,
    memory,
    frontend: @program.AudioFrontendProgram::new(sources.frontend),
    tail: @program.AudioTailProgram::new(sources.tail),
    stages: [],
    modules: [],
    chain: None,
    decoder: None,
    parameters: [],
    front_weights: None,
    tail_weights: None,
    started: false,
    ready: false,
  }
}

///|
pub fn CheckpointAudioDecoder::memory(
  self : Self,
) -> @program.AudioDecoderMemoryPlan {
  self.memory
}

///|
/// All AOT binaries are supplied by an offline loader. There is no compilation
/// here. Retain this owner on any partial failure and close it deterministically.
pub fn CheckpointAudioDecoder::prepare(
  self : Self,
  context : @device.Context,
  stream : @device.Stream,
  weights : @startup.WeightLease,
  load : (String) -> @device.Module raise,
) -> Unit raise {
  guard !self.started else { fail("AudioVAE bootstrap already started") }
  self.started = true
  fn module(name : String) -> @device.Module raise {
    let value = load(name)
    self.modules.push(value)
    value
  }
  self.frontend.prepare(
    context,
    module("audio_front"),
    max_scratch_bytes=self.memory.frontend_bytes,
  )
  for i in 0..<7 {
    let stage = @program.AudioDecodeStage::new(
      self.sources.upsample[i],
      self.sources.amp[i],
    )
    self.stages.push(stage)
    stage.prepare(
      context,
      module("audio_up_\{i}"),
      module("audio_amp_\{i}"),
      max_workspace_bytes=self.memory.stage_bytes[i],
    )
    self.parameters.push(
      (
        @vae.bind_audio_decoder_amp_parameter_layout(stage_index=i),
        weights.with_allocation(1 + i * 2, a => a),
        weights.with_allocation(2 + i * 2, a => a),
      ),
    )
  }
  self.tail.prepare(
    context,
    module("audio_tail"),
    max_scratch_bytes=self.memory.tail_bytes,
  )
  let chain = @program.AudioDecodeChain::allocate(
    context,
    self.stages[:],
    max_intermediate_bytes=self.memory.chain_bytes,
  )
  self.chain = Some(chain)
  let decoder = @program.AudioDecoder::allocate(
    context,
    self.frontend,
    chain,
    self.tail,
    max_bridge_bytes=self.memory.bridge_bytes,
  )
  self.decoder = Some(decoder)
  let front = weights.with_allocation(0, a => a)
  let tail = weights.with_allocation(15, a => a)
  self.front_weights = Some(front)
  self.tail_weights = Some(tail)
  decoder.prepare_weights(
    context,
    stream,
    self.parameters[:],
    frontend_weights=front,
    tail_weights=tail,
    max_cache_bytes=self.memory.cache_bytes,
  )
  self.ready = true
}

///|
/// Bind one raw channel-major, destandardized F32 latent input. The caller owns
/// the request before this call so partial queue preparation remains releasable.
/// This entry point does not substitute for full denoiser-to-AudioInput joining.
pub fn CheckpointAudioDecoder::prepare_request(
  self : Self,
  request : @program.AudioDecodeRequest,
  context : @device.Context,
  stream : @device.Stream,
  input : @device.Allocation,
) -> Unit raise {
  guard self.ready &&
    self.decoder is Some(decoder) &&
    self.front_weights is Some(front) &&
    self.tail_weights is Some(tail) &&
    input.byte_count() == self.memory.input_bytes else {
    fail("AudioVAE decoder not ready or input geometry differs")
  }
  request.prepare(
    context,
    stream,
    decoder,
    self.parameters[:],
    input~,
    input_offset=0L,
    frontend_weights=front,
    tail_weights=tail,
    max_output_bytes=self.memory.output_bytes,
  )
}

///|
pub fn CheckpointAudioDecoder::close(self : Self) -> Unit raise {
  self.ready = false
  self.started = true
  if self.decoder is Some(value) {
    value.close()
    self.decoder = None
  }
  if self.chain is Some(value) {
    value.close()
    self.chain = None
  }
  self.tail.close()
  while !self.stages.is_empty() {
    let index = self.stages.length() - 1
    self.stages[index].close()
    self.stages.remove(index) |> ignore
  }
  self.frontend.close()
  while !self.modules.is_empty() {
    let index = self.modules.length() - 1
    self.modules[index].close()
    self.modules.remove(index) |> ignore
  }
  self.parameters.clear()
  self.front_weights = None
  self.tail_weights = None
}
