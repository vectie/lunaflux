///|
/// Bind structural target extents to the resolved H3 request. Conditioning
/// row counts come from completed encoders; raw media dimensions are not used
/// to guess the encoder output. Their contents/order must follow segments().
pub fn Sequence::from_request(
  model : @joint_diffusion_plan.JointDiffusionPlan,
  request : @joint_diffusion_plan.JointDiffusionRequestShape,
  text_rows~ : Int,
  conditioning~ : ArrayView[ConditionBlock],
  max_rows~ : Int,
) -> Self raise SequenceError {
  let shape = model.shapes()
  let resolved = shape.resolve_request(
    height=request.height(),
    width=request.width(),
    requested_frames=request.frames(),
    inference_steps=request.inference_steps(),
  ) catch {
    _ => raise InvalidShape
  }
  guard resolved == request &&
    request.video_latent_frames() % shape.video_patch_temporal() == 0 &&
    request.video_latent_height() % shape.video_patch_height() == 0 &&
    request.video_latent_width() % shape.video_patch_width() == 0 else {
    raise InvalidShape
  }
  let video_rows = for
    dim in [
      request.video_latent_frames() / shape.video_patch_temporal(),
      request.video_latent_height() / shape.video_patch_height(),
      request.video_latent_width() / shape.video_patch_width(),
    ]
    rows = 1L {
    guard dim > 0 && rows <= max_rows.to_int64() / dim.to_int64() else {
      raise RowLimitExceeded
    }
    continue rows * dim.to_int64()
  } nobreak {
    rows
  }
  Sequence::new(
    hidden_width=model.denoiser().hidden_size(),
    text_rows~,
    video_rows=video_rows.to_int(),
    audio_rows=request.audio_token_rows(),
    conditioning~,
    max_rows~,
  )
}
