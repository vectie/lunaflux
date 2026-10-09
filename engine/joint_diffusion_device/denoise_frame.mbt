///|
/// Prepare one joint frame: all model launches first, then both in-place RF
/// updates on the same stream. The model launches must produce both velocity
/// buffers without mutating the latent states, and already bind this step's
/// timesteps. This constructor does not manufacture missing model launches.
/// All arguments/lists are prepared before native queue ownership is created;
/// if preparation fails, no native queue needs releasing.
pub fn KernelFrame::denoise(
  context : @device.Context,
  stream : @device.Stream,
  storage : LatentStorage,
  prediction : ArrayView[@device.OrderedKernelLaunch],
  step~ : Int,
  video_update~ : @device.Function,
  audio_update~ : @device.Function,
  video_block_size~ : Int,
  audio_block_size~ : Int,
  policy? : @device.OrderedKernelExecutionPolicy = OrderedKernelEagerOnly,
) -> Self raise {
  guard prediction.length() > 0 else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  let video = storage.flow_launch(
    video_update,
    step~,
    latent=VideoLatent,
    block_size=video_block_size,
  )
  let audio = storage.flow_launch(
    audio_update,
    step~,
    latent=AudioLatent,
    block_size=audio_block_size,
  )
  let launches = prediction.to_owned()
  launches.push(video)
  launches.push(audio)
  KernelFrame::new(context, stream, launches, policy~)
}
