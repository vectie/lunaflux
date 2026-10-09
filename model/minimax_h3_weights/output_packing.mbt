///|
pub struct OutputWeightPacking {
  norm_offset : Int64
  adaln_offset : Int64
  adaln_bias_offset : Int64
  video_offset : Int64
  video_bias_offset : Int64
  audio_offset : Int64
  audio_bias_offset : Int64
  byte_count : Int64
  copies : ReadOnlyArray[BlockWeightCopy]
}

///|
/// Pack the final block without dtype conversion: BF16 norm/AdaLN and F32
/// media heads. Names are checkpoint names, not backend module aliases.
pub fn output_weight_packing(
  model : @joint_diffusion_plan.JointDiffusionPlan,
  max_bytes~ : Int64,
) -> OutputWeightPacking raise BlockPackingError {
  let hidden = model.denoiser().hidden_size().to_int64()
  let conditioning = model
    .denoiser()
    .final_adaln_conditioning_dimension()
    .to_int64()
  let geometry = model.shapes()
  guard max_bytes > 0L && hidden <= max_bytes / 4L / conditioning else {
    raise BudgetExceeded
  }
  let mut video = geometry.video_latent_channels().to_int64()
  for
    d in [
      geometry.video_patch_temporal(),
      geometry.video_patch_height(),
      geometry.video_patch_width(),
    ] {
    guard video <= max_bytes / d.to_int64() else { raise BudgetExceeded }
    video *= d.to_int64()
  }
  let audio = geometry.audio_latent_channels().to_int64()
  guard video <= max_bytes / 4L / hidden && audio <= max_bytes / 4L / hidden else {
    raise BudgetExceeded
  }
  let copies : Array[BlockWeightCopy] = []
  fn add(
    name : String,
    size : Int64,
    offset : Int64,
  ) -> Int64 raise BlockPackingError {
    guard size <= max_bytes - offset else { raise BudgetExceeded }
    copies.push({
      tensor_name: name,
      source_offset: 0L,
      destination_offset: offset,
      byte_count: size,
    })
    offset + size
  }
  let adaln_offset = add("norm_out.norm.weight", hidden * 2L, 0L)
  let adaln_bias_offset = add(
    "norm_out.linear.weight",
    hidden * conditioning * 4L,
    adaln_offset,
  )
  let video_offset = add("norm_out.linear.bias", hidden * 4L, adaln_bias_offset)
  let video_bias_offset = add(
    "proj_out.weight",
    video * hidden * 4L,
    video_offset,
  )
  let audio_offset = add("proj_out.bias", video * 4L, video_bias_offset)
  let audio_bias_offset = add(
    "audio_proj_out.weight",
    audio * hidden * 4L,
    audio_offset,
  )
  let byte_count = add("audio_proj_out.bias", audio * 4L, audio_bias_offset)
  {
    norm_offset: 0L,
    adaln_offset,
    adaln_bias_offset,
    video_offset,
    video_bias_offset,
    audio_offset,
    audio_bias_offset,
    byte_count,
    copies: ReadOnlyArray::from_array(copies),
  }
}
