///|
fn bind_latents(
  assessment : JointVaeDecodeAssessment,
  component : @joint_diffusion_plan.JointDiffusionComponent,
  values : ArrayView[Double],
) -> JointVaeLatentBuffer raise JointVaeDecodeReferenceError {
  let expected = match component {
    VideoVae => assessment.geometry.video_latent_elements
    AudioVae => assessment.geometry.audio_latent_elements
    _ => raise InvalidPlan
  }
  if values.length().to_int64() != expected {
    raise InvalidLatentElementCount(component)
  }
  for index, value in values {
    if value.is_nan() || value.is_inf() {
      raise NonFiniteLatent(component, index)
    }
  }
  {
    component,
    plan_digest: assessment.geometry.plan_digest,
    values: ReadOnlyArray::from_array(values),
  }
}

///|
pub fn bind_video_latents(
  assessment : JointVaeDecodeAssessment,
  values : ArrayView[Double],
) -> JointVaeLatentBuffer raise JointVaeDecodeReferenceError {
  bind_latents(assessment, VideoVae, values)
}

///|
pub fn bind_audio_latents(
  assessment : JointVaeDecodeAssessment,
  values : ArrayView[Double],
) -> JointVaeLatentBuffer raise JointVaeDecodeReferenceError {
  bind_latents(assessment, AudioVae, values)
}

///|
fn require_decode(
  assessment : JointVaeDecodeAssessment,
  buffer : JointVaeLatentBuffer,
  component : @joint_diffusion_plan.JointDiffusionComponent,
) -> Unit raise JointVaeDecodeReferenceError {
  if buffer.component != component ||
    buffer.plan_digest != assessment.geometry.plan_digest {
    raise LatentIdentityMismatch(component)
  }
  let gaps = match component {
    VideoVae => assessment.video_gaps
    AudioVae => assessment.audio_gaps
    _ => raise InvalidPlan
  }
  guard gaps.get(0) is Some(first) else { raise InvalidPlan }
  raise DecodeSemanticUnavailable(component, first)
}

///|
/// Fails closed because the generic plan does not yet prove exact video decode
/// execution semantics.
pub fn decode_one_video_stage(
  assessment : JointVaeDecodeAssessment,
  latent : JointVaeLatentBuffer,
) -> ReadOnlyArray[Double] raise JointVaeDecodeReferenceError {
  require_decode(assessment, latent, VideoVae)
  abort("unreachable")
}

///|
/// Fails closed because the generic plan does not yet prove exact audio decode
/// execution semantics. Artifact materialization alone is insufficient.
pub fn decode_one_audio_stage(
  assessment : JointVaeDecodeAssessment,
  latent : JointVaeLatentBuffer,
) -> ReadOnlyArray[Double] raise JointVaeDecodeReferenceError {
  require_decode(assessment, latent, AudioVae)
  abort("unreachable")
}
