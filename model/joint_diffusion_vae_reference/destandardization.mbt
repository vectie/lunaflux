///|
fn finite_f32(value : Float) -> Bool {
  (value.reinterpret_as_uint() & 0x7f800000U) != 0x7f800000U
}

///|
fn destandardization_digest(
  plan_digest : @spec.PlanDigest,
  request : @joint_diffusion_plan.JointDiffusionRequestShape,
  means : ReadOnlyArray[Float],
  standard_deviations : ReadOnlyArray[Float],
) -> JointVaeLatentDestandardizationDigest {
  let canonical = StringBuilder()
  canonical <+ "lunaflux.joint-vae-video-latent-destandardization.v1"
  canonical <+ "|plan=\{plan_digest.as_hex()}"
  canonical <+ "|layout=b-c-f-h-w|storage=f32-input-std-mean-output"
  canonical <+ "|frames=\{request.video_latent_frames()}"
  canonical <+ "|height=\{request.video_latent_height()}"
  canonical <+ "|width=\{request.video_latent_width()}"
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
/// Authenticates the exact request geometry and ordered F32 coefficient table
/// for decode-side `latent * standard_deviation + mean`.
pub fn bind_video_latent_destandardization(
  plan : @joint_diffusion_plan.JointDiffusionPlan,
  request : @joint_diffusion_plan.JointDiffusionRequestShape,
  means~ : ArrayView[Float],
  standard_deviations~ : ArrayView[Float],
) -> JointVaeLatentDestandardization raise JointVaeDecodeReferenceError {
  if !request_matches_plan(plan, request) {
    raise InvalidRequest
  }
  let channels = plan.shapes().video_latent_channels()
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
  let elements_per_channel = checked_product([
    request.video_latent_frames().to_int64(),
    request.video_latent_height().to_int64(),
    request.video_latent_width().to_int64(),
  ])
  let element_count = checked_product([
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
    layout: BatchChannelFrameHeightWidth,
    batch_count: 1,
    channel_count: channels,
    elements_per_channel,
    element_count,
    means: exact_means,
    standard_deviations: exact_standard_deviations,
    digest: destandardization_digest(
      plan.plan_digest(),
      request,
      exact_means,
      exact_standard_deviations,
    ),
  }
}

///|
/// Scalar reference with an explicit F32 rounding boundary after multiply and
/// after add. This is not device execution authority.
pub fn destandardize_video_latents(
  contract : JointVaeLatentDestandardization,
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
    let channel = index.to_int64() / contract.elements_per_channel
    let scaled = value * contract.standard_deviations[channel.to_int()]
    output.push(scaled + contract.means[channel.to_int()])
  }
  ReadOnlyArray::from_array(output)
}
