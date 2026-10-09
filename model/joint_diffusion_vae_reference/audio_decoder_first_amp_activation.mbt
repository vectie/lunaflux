///|
pub fn bind_audio_decoder_first_amp_activation(
  plan : @joint_diffusion_plan.JointDiffusionPlan,
  request : @joint_diffusion_plan.JointDiffusionRequestShape,
  decoder_batch_count~ : Int,
  alpha_name~ : String,
  beta_name~ : String,
  upsample_filter_name~ : String,
  downsample_filter_name~ : String,
) -> JointVaeAudioDecoderFirstAmpActivation raise JointVaeDecodeReferenceError {
  if !request_matches_plan(plan, request) ||
    decoder_batch_count <= 0 ||
    !valid_tensor_name(alpha_name) ||
    !valid_tensor_name(beta_name) ||
    !valid_tensor_name(upsample_filter_name) ||
    !valid_tensor_name(downsample_filter_name) ||
    upsample_filter_name == downsample_filter_name {
    raise InvalidProjectionTensorIdentity
  }
  let channels = plan.shapes().audio_vae_decoder_dimension() / 2
  let time = request.audio_latent_frames() * 5
  let upsample_time = time * 2
  let elements = checked_product([
    decoder_batch_count.to_int64(),
    channels.to_int64(),
    time.to_int64(),
  ])
  let intermediate = checked_product([
    decoder_batch_count.to_int64(),
    channels.to_int64(),
    upsample_time.to_int64(),
  ])
  let canonical = StringBuilder()
  canonical <+ "lunaflux.joint-vae-audio-decoder-first-amp-activation.v1"
  canonical <+
    "|plan=\{plan.plan_digest().as_hex()}|layout=b-c-t|storage=f32-input-alpha-beta-up-filter-down-filter-workspaces-output"
  canonical <+
    "|batch=\{decoder_batch_count}|channels=\{channels}|time=\{time}:\{upsample_time}|ratio=2|kernel=12"
  canonical <+
    "|alpha=\{alpha_name}|beta=\{beta_name}|up=\{upsample_filter_name}|down=\{downsample_filter_name}"
  {
    plan_digest: plan.plan_digest(),
    request_shape: request,
    decoder_batch_count,
    channel_count: channels,
    time_count: time,
    upsample_time,
    ratio: 2,
    kernel_size: 12,
    storage: Float32InputAlphaBetaUpsampleFilterDownsampleFilterWorkspacesOutput,
    layout: BatchChannelTime,
    alpha_name,
    beta_name,
    upsample_filter_name,
    downsample_filter_name,
    element_count: elements,
    intermediate_element_count: intermediate,
    digest: {
      hex: @crypto.bytes_to_hex_string(
        @crypto.sha256(@utf8.encode(canonical.to_string())),
      ),
    },
  }
}

///|
pub fn execute_audio_decoder_first_amp_activation(
  contract : JointVaeAudioDecoderFirstAmpActivation,
  input : ArrayView[Float],
  alpha : ArrayView[Float],
  beta : ArrayView[Float],
  upsample_filter : ArrayView[Float],
  downsample_filter : ArrayView[Float],
) -> ReadOnlyArray[Float] raise JointVaeDecodeReferenceError {
  execute_audio_decoder_amp_activation_values(
    channel_count=contract.channel_count,
    time_count=contract.time_count,
    upsample_time=contract.upsample_time,
    element_count=contract.element_count,
    intermediate_element_count=contract.intermediate_element_count,
    input,
    alpha,
    beta,
    upsample_filter,
    downsample_filter,
  )
}
