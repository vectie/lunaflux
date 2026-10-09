///|
fn audio_destandardization_digest(
  plan_digest : @spec.PlanDigest,
  request : @joint_diffusion_plan.JointDiffusionRequestShape,
  batch_count : Int,
  means : ReadOnlyArray[Float],
  standard_deviations : ReadOnlyArray[Float],
) -> JointVaeLatentDestandardizationDigest {
  let canonical = StringBuilder()
  canonical <+ "lunaflux.joint-vae-audio-latent-destandardization.v1"
  canonical <+ "|plan=\{plan_digest.as_hex()}"
  canonical <+ "|layout=b-c-t|storage=f32-input-std-mean-output"
  canonical <+ "|batch=\{batch_count}|time=\{request.audio_latent_frames()}"
  for index in 0..<means.length() {
    canonical <+ "|channel=\{index}"
    canonical <+ ":std=\{standard_deviations[index].reinterpret_as_uint()}"
    canonical <+ ":mean=\{means[index].reinterpret_as_uint()}"
  }
  {
    hex: @crypto.bytes_to_hex_string(
      @crypto.sha256(@utf8.encode(canonical.to_string())),
    ),
  }
}

///|
pub fn bind_audio_latent_destandardization(
  plan : @joint_diffusion_plan.JointDiffusionPlan,
  request : @joint_diffusion_plan.JointDiffusionRequestShape,
  batch_count~ : Int,
  means~ : ArrayView[Float],
  standard_deviations~ : ArrayView[Float],
) -> JointVaeAudioLatentDestandardization raise JointVaeDecodeReferenceError {
  if !request_matches_plan(plan, request) || batch_count <= 0 {
    raise InvalidRequest
  }
  let channels = plan.shapes().audio_latent_channels()
  if means.length() != channels || standard_deviations.length() != channels {
    raise InvalidDestandardizationCoefficients
  }
  for index in 0..<channels {
    if !finite_f32(means[index]) ||
      !finite_f32(standard_deviations[index]) ||
      standard_deviations[index] <= 0.0F {
      raise InvalidDestandardizationCoefficients
    }
  }
  let elements_per_channel = request.audio_latent_frames().to_int64()
  let element_count = checked_product([
    batch_count.to_int64(),
    channels.to_int64(),
    elements_per_channel,
  ])
  let exact_means = ReadOnlyArray::from_array(means.to_owned())
  let exact_standard_deviations = ReadOnlyArray::from_array(
    standard_deviations.to_owned(),
  )
  {
    plan_digest: plan.plan_digest(),
    request_shape: request,
    storage: Float32InputScaleMeanOutput,
    layout: BatchChannelTime,
    batch_count,
    channel_count: channels,
    elements_per_channel,
    element_count,
    means: exact_means,
    standard_deviations: exact_standard_deviations,
    digest: audio_destandardization_digest(
      plan.plan_digest(),
      request,
      batch_count,
      exact_means,
      exact_standard_deviations,
    ),
  }
}

///|
pub fn destandardize_audio_latents(
  contract : JointVaeAudioLatentDestandardization,
  normalized : ArrayView[Float],
) -> ReadOnlyArray[Float] raise JointVaeDecodeReferenceError {
  if normalized.length().to_int64() != contract.element_count {
    raise InvalidDestandardizationInput
  }
  let output : Array[Float] = []
  for index, value in normalized {
    if !finite_f32(value) {
      raise InvalidDestandardizationInput
    }
    let channel = index.to_int64() /
      contract.elements_per_channel %
      contract.channel_count.to_int64()
    let scaled = value * contract.standard_deviations[channel.to_int()]
    output.push(scaled + contract.means[channel.to_int()])
  }
  ReadOnlyArray::from_array(output)
}
