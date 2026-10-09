///|
fn[T] prediction_stages(
  timestep : ArrayView[T],
  inputs : ArrayView[T],
  denoiser : ArrayView[T],
) -> Array[T] raise @device.DeviceError {
  guard !timestep.is_empty() && !inputs.is_empty() && !denoiser.is_empty() else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  let launches = timestep.to_owned()
  launches.append(inputs)
  launches.append(denoiser)
  launches
}

///|
/// Prepare a whole denoising iteration with one completion boundary. Timestep
/// production precedes input projection/packing, then the complete denoiser
/// and both RF updates. Lists are immutable prepared bindings for this step;
/// all inputs must read the old latent state and no earlier stage may update it.
pub fn KernelFrame::denoise_staged(
  context : @device.Context,
  stream : @device.Stream,
  storage : LatentStorage,
  timestep : ArrayView[@device.OrderedKernelLaunch],
  inputs : ArrayView[@device.OrderedKernelLaunch],
  denoiser : ArrayView[@device.OrderedKernelLaunch],
  step~ : Int,
  video_update~ : @device.Function,
  audio_update~ : @device.Function,
  video_block_size~ : Int,
  audio_block_size~ : Int,
  policy? : @device.OrderedKernelExecutionPolicy = OrderedKernelEagerOnly,
) -> Self raise {
  let prediction = prediction_stages(timestep, inputs, denoiser)
  KernelFrame::denoise(
    context,
    stream,
    storage,
    prediction,
    step~,
    video_update~,
    audio_update~,
    video_block_size~,
    audio_block_size~,
    policy~,
  )
}
