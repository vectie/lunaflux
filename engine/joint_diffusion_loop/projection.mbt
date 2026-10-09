///|
pub(all) suberror ProjectionPlanError {
  ModelMismatch
  ByteLimitExceeded
}

///|
/// F32 packed rows and row-major [output,input] weights, separate F32 bias,
/// BF16 output. This contract is shared by video patches and stereo audio rows.
pub struct LatentProjectionShape {
  rows : Int64
  input_width : Int
  output_width : Int
  weight_bytes : Int64
  bias_bytes : Int64
  output : StorageRegion
}

///|
pub struct LatentProjectionPlan {
  priv video : LatentProjectionShape
  priv audio : LatentProjectionShape
  priv bytes : Int64
}

///|
/// Derive projection workspaces from this arena's exact request. The budget
/// covers both BF16 outputs including alignment, not weights or transformer
/// activations. No device policy or family-specific branching occurs here.
pub fn LatentStoragePlan::projections(
  self : Self,
  model : @joint_diffusion_plan.JointDiffusionPlan,
  max_bytes~ : Int64,
) -> LatentProjectionPlan raise ProjectionPlanError {
  guard model.plan_digest() == self.model_digest else { raise ModelMismatch }
  guard max_bytes > 0L else { raise ByteLimitExceeded }
  let shape = model.shapes()
  let video_width = bounded_product(
    [
      shape.video_latent_channels(),
      shape.video_patch_temporal(),
      shape.video_patch_height(),
      shape.video_patch_width(),
    ],
    2147483647L,
  ) catch {
    _ => raise ByteLimitExceeded
  }
  let audio_width = shape.audio_latent_channels()
  let hidden = model.denoiser().hidden_size()
  let video_rows = self.region(VideoState).byte_count / 4L / video_width
  let audio_rows = self.request.audio_token_rows().to_int64()
  let video = projection_shape(
    video_rows,
    video_width.to_int(),
    hidden,
    0L,
    max_bytes,
  )
  let end = video.output.byte_count
  let padding = (256L - end % 256L) % 256L
  guard padding <= max_bytes - end else { raise ByteLimitExceeded }
  let audio = projection_shape(
    audio_rows,
    audio_width,
    hidden,
    end + padding,
    max_bytes,
  )
  { video, audio, bytes: audio.output.offset + audio.output.byte_count, }
}

///|
fn projection_shape(
  rows : Int64,
  input : Int,
  output : Int,
  offset : Int64,
  ceiling : Int64,
) -> LatentProjectionShape raise ProjectionPlanError {
  guard rows > 0L &&
    input > 0 &&
    output > 0 &&
    offset <= ceiling &&
    rows <= (ceiling - offset) / 2L / output.to_int64() &&
    input.to_int64() <= 9223372036854775807L / 4L / output.to_int64() else {
    raise ByteLimitExceeded
  }
  {
    rows,
    input_width: input,
    output_width: output,
    weight_bytes: input.to_int64() * output.to_int64() * 4L,
    bias_bytes: output.to_int64() * 4L,
    output: { offset, byte_count: rows * output.to_int64() * 2L, },
  }
}

///|
pub fn LatentProjectionPlan::byte_count(self : Self) -> Int64 {
  self.bytes
}

///|
pub fn LatentProjectionPlan::shape(
  self : Self,
  latent : @joint_diffusion_plan.JointDiffusionLatentKind,
) -> LatentProjectionShape {
  match latent {
    VideoLatent => self.video
    AudioLatent => self.audio
  }
}
