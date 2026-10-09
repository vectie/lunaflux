///|
/// Typed full-model preparation: complete encoded/reference row assembly is
/// always appended after its producers, then exact scheduled predictions and
/// both decoders. External weights, modules, latents and metadata remain borrowed
/// and must outlive this request. Initial binding failure transfers no stages.
pub fn Request::prepare_model(
  self : Self,
  context : @device.Context,
  stream : @device.Stream,
  plan : @joint_diffusion_loop.PreparedPlan,
  storage : @joint_diffusion_device.LatentStorage,
  producers : ArrayView[@joint_diffusion_execution.PreparedStage],
  conditioning : @blocks.ConditioningProgram,
  metadata : @blocks.PredictionMetadata,
  variants : ArrayView[@blocks.PredictionVariant],
  input : @blocks.InputProgram,
  rotary : @joint_diffusion_device.RotaryBindings,
  weights : @blocks.PredictionWeights,
  video : @video.VideoDecodeRequest,
  audio : @blocks.AudioDecodeRequest,
  timestep_embeddings~ : @device.Allocation,
  timestep_embeddings_offset~ : Int64,
  parameters~ : @device.Allocation,
  parameter_offset~ : Int64,
  video_update~ : @device.Function,
  audio_update~ : @device.Function,
  video_block_size~ : Int,
  audio_block_size~ : Int,
) -> Unit raise {
  guard !self.attempted &&
    !self.closed &&
    !producers.is_empty() &&
    metadata.matches_plan(plan) &&
    storage.matches_prepared(plan) else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  let predictions = metadata.bind_conditioned(
    variants,
    input,
    storage,
    rotary,
    weights,
    conditioning,
    timestep_embeddings~,
    timestep_embeddings_offset~,
    parameters~,
    parameter_offset~,
  )
  let stages = producers.to_owned()
  stages.push(@joint_diffusion_execution.PreparedStage::own(conditioning))
  self.prepare(
    context,
    stream,
    plan,
    storage,
    stages,
    predictions[:],
    video,
    audio,
    video_update~,
    audio_update~,
    video_block_size~,
    audio_block_size~,
  )
}
