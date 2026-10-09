///|
pub struct InputWeightPacking {
  video_offset : Int64
  video_bias_offset : Int64
  audio_offset : Int64
  audio_bias_offset : Int64
  byte_count : Int64
  copies : ReadOnlyArray[BlockWeightCopy]
}

///|
/// Pack F32 media input projections with separate hidden-width bias vectors.
pub fn input_weight_packing(
  model : @joint_diffusion_plan.JointDiffusionPlan,
  max_bytes~ : Int64,
) -> InputWeightPacking raise BlockPackingError {
  let shape = model.shapes()
  let h = model.denoiser().hidden_size().to_int64()
  let mut video = shape.video_latent_channels().to_int64()
  guard max_bytes > 0L else { raise BudgetExceeded }
  for
    d in [
      shape.video_patch_temporal(),
      shape.video_patch_height(),
      shape.video_patch_width(),
    ] {
    guard video <= max_bytes / d.to_int64() else { raise BudgetExceeded }
    video *= d.to_int64()
  }
  let audio = shape.audio_latent_channels().to_int64()
  guard video <= max_bytes / 4L / h && audio <= max_bytes / 4L / h else {
    raise BudgetExceeded
  }
  let names = [
    "proj_in.weight", "proj_in.bias", "audio_proj_in.weight", "audio_proj_in.bias",
  ]
  let sizes = [video * h * 4L, h * 4L, audio * h * 4L, h * 4L]
  let copies : Array[BlockWeightCopy] = []
  let mut offset = 0L
  for i, size in sizes {
    guard size <= max_bytes - offset else { raise BudgetExceeded }
    copies.push({
      tensor_name: names[i],
      source_offset: 0L,
      destination_offset: offset,
      byte_count: size,
    })
    offset += size
  }
  {
    video_offset: 0L,
    video_bias_offset: copies[1].destination_offset,
    audio_offset: copies[2].destination_offset,
    audio_bias_offset: copies[3].destination_offset,
    byte_count: offset,
    copies: ReadOnlyArray::from_array(copies),
  }
}
