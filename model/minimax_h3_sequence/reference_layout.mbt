///|
/// Raw VAE [C,T,H,W] to packed [T/pt,H/ph,W/pw,C,pt,ph,pw].
/// Dimensions are latent dimensions, not decoded pixel/frame dimensions.
pub fn reference_video_layout(
  model : @joint_diffusion_plan.JointDiffusionPlan,
  frames~ : Int,
  height~ : Int,
  width~ : Int,
  expected_rows~ : Int,
  max_elements~ : Int64,
) -> @dense_permutation_plan.Permutation raise SequenceError {
  let shape = model.shapes()
  let pt = shape.video_patch_temporal()
  let ph = shape.video_patch_height()
  let pw = shape.video_patch_width()
  guard frames > 0 &&
    height > 0 &&
    width > 0 &&
    expected_rows > 0 &&
    frames % pt == 0 &&
    height % ph == 0 &&
    width % pw == 0 else {
    raise InvalidShape
  }
  let plan = @dense_permutation_plan.Permutation::new(
    [
      shape.video_latent_channels(),
      frames / pt,
      pt,
      height / ph,
      ph,
      width / pw,
      pw,
    ],
    [1, 3, 5, 0, 2, 4, 6],
    max_elements~,
  ) catch {
    _ => raise InvalidShape
  }
  // Divide the checked product rather than multiplying unbounded dimensions.
  let rows = plan.element_count() /
    shape.video_latent_channels().to_int64() /
    pt.to_int64() /
    ph.to_int64() /
    pw.to_int64()
  guard rows == expected_rows.to_int64() else { raise InvalidConditioning }
  plan
}

///|
/// Raw VAE [streams,channels,frames] to packed [streams,frames,channels].
pub fn reference_audio_layout(
  model : @joint_diffusion_plan.JointDiffusionPlan,
  frames~ : Int,
  expected_rows~ : Int,
  max_elements~ : Int64,
) -> @dense_permutation_plan.Permutation raise SequenceError {
  let shape = model.shapes()
  guard frames > 0 &&
    expected_rows > 0 &&
    shape.audio_latent_streams().to_int64() * frames.to_int64() ==
    expected_rows.to_int64() else {
    raise InvalidConditioning
  }
  @dense_permutation_plan.Permutation::new(
    [shape.audio_latent_streams(), shape.audio_latent_channels(), frames],
    [0, 2, 1],
    max_elements~,
  ) catch {
    _ => raise InvalidShape
  }
}
