///|
/// Geometry-only startup query. A matching shape does not prove that two
/// requests share latent storage; prepare_joint_request binds the actual owner.
pub fn CheckpointAudioDecoder::matches_plan(
  self : Self,
  plan : @loop.PreparedPlan,
) -> Bool {
  let time = plan.request().audio_latent_frames()
  time > 0 &&
  self.sources.frontend.batch == 2 &&
  self.sources.frontend.time == time &&
  self.sources.tail.batch == 2 &&
  self.sources.tail.time.to_int64() == time.to_int64() * 800L &&
  self.memory.input_bytes == plan.audio_elements() * 4L &&
  self.memory.output_bytes == time.to_int64() * 800L * 8L
}

///|
/// Aggregate decoder scratch/cache, two AudioInputProgram regions and one
/// waveform output. Excludes borrowed checkpoint/module/statistic allocations
/// and the joint LatentStorage (which is counted by its own owner).
pub fn CheckpointAudioDecoder::joint_workspace_bytes(self : Self) -> Int64 {
  self.memory.total_bytes + self.memory.input_bytes
}

///|
/// Bind the real packed AudioState from a joint denoiser through permutation
/// and checkpoint destandardization, then the complete AudioVAE. No host copy,
/// queue wait or latent fabrication occurs here. The caller owns request and
/// input_program before this call, including on partial preparation failure.
pub fn CheckpointAudioDecoder::prepare_joint_request(
  self : Self,
  request : @program.AudioDecodeRequest,
  context : @device.Context,
  stream : @device.Stream,
  plan : @loop.PreparedPlan,
  storage : @latent.LatentStorage,
  input_program : @program.AudioInputProgram,
) -> Unit raise {
  guard self.ready &&
    self.decoder is Some(decoder) &&
    self.front_weights is Some(front) &&
    self.tail_weights is Some(tail) &&
    self.matches_plan(plan) &&
    storage.matches_prepared(plan) else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  storage.with_region(AudioState, (input, offset, bytes) => {
    guard bytes == self.memory.input_bytes else {
      raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
    }
    request.prepare(
      context,
      stream,
      decoder,
      self.parameters[:],
      input~,
      input_offset=offset,
      frontend_weights=front,
      tail_weights=tail,
      max_output_bytes=self.memory.output_bytes,
      input_program~,
      prepared_plan=plan,
    )
  })
}
