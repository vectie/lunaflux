///|
pub struct VideoInputSource {
  plan : @joint_diffusion_loop.PreparedPlan
  source : Bytes
  symbols : ReadOnlyArray[String]
  permutation : @dense_permutation_plan.Permutation
  elements_per_channel : Int64
  value_bytes : Int64
}

///|
/// Final denoiser rows -> F32 [24,T,H,W] -> checkpoint destandardization.
/// The pure inverse of the request's input packing avoids a second independently
/// maintained shape formula. This emits AOT source; it never compiles at runtime.
pub fn build_video_input(
  plan : @joint_diffusion_loop.PreparedPlan,
) -> VideoInputSource raise {
  let request = plan.request()
  let spatial = request.video_latent_frames().to_int64() *
    request.video_latent_height().to_int64() *
    request.video_latent_width().to_int64()
  guard spatial > 0L &&
    plan.video_elements() / spatial == 24L &&
    plan.video_elements() % spatial == 0L else {
    raise InvalidShape
  }
  let permutation = plan.video_pack().inverse()
  let affine = @affine_modulation_cuda_source.f32_channel_major_destandardization_source(
    batch_count=1,
    channels=24,
    elements_per_channel=spatial,
    function_symbol="lunaflux_video_destandardize_v1",
  )
  {
    plan,
    source: @cuda_translation_unit.compose([
      @dense_permutation_cuda_source.render(permutation, block_size=256),
      affine.source_bytes(),
    ]),
    symbols: [
      "lunaflux_dense_permutation_u32_v1", "lunaflux_video_destandardize_v1",
    ],
    permutation,
    elements_per_channel: spatial,
    value_bytes: plan.video_elements() * 4L,
  }
}
