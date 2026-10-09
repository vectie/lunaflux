///|
/// Authenticates released `decoder.ups.1.0`, immediately after the first
/// three-way AMP residual average. The following `resblocks.3.*` activation
/// remains a separate operation.
pub fn bind_audio_decoder_second_upsample(
  metadata : @minimax_h3_spec.MiniMaxH3ModelMetadata,
  config : @minimax_h3_config.MiniMaxH3AudioVaeConfig,
  request : @joint_diffusion_plan.JointDiffusionRequestShape,
) -> @joint_diffusion_vae_reference.JointVaeAudioDecoderUpsampleTransition raise MiniMaxH3VaeReferenceError {
  let plan = @minimax_h3.build(metadata) catch { error => raise Plan(error) }
  let schema = @minimax_h3_weights.MiniMaxH3AudioVaeSchema::official_v1()
  if config.architecture_schema_version() != 1 ||
    config.decoder_dimension() != 1024 ||
    config.decoder_rates().length() != 7 ||
    config.decoder_kernel_sizes().length() != 7 ||
    config.decoder_rates()[1] != 5 ||
    config.decoder_kernel_sizes()[1] != 9 ||
    config.output_channels() != 2 ||
    !schema.matches_tensor("decoder.ups.1.0.weight_g", Float32, [512L, 1L, 1L]) ||
    !schema.matches_tensor("decoder.ups.1.0.weight_v", Float32, [512L, 256L, 9L]) ||
    !schema.matches_tensor("decoder.ups.1.0.bias", Float32, [256L]) {
    raise InvalidAudioVaeConfig
  }
  let input_time64 = request.audio_latent_frames().to_int64() * 5L
  let output_time64 = input_time64 * 5L
  let input_time = input_time64.to_int()
  let output_time = output_time64.to_int()
  if input_time.to_int64() != input_time64 ||
    output_time.to_int64() != output_time64 {
    raise InvalidAudioVaeConfig
  }
  @joint_diffusion_vae_reference.bind_audio_decoder_upsample_transition(
    plan,
    request,
    stage_index=1,
    decoder_batch_count=2,
    input_channels=512,
    output_channels=256,
    input_time~,
    output_time~,
    kernel_size=9,
    stride=5,
    padding=2,
    weight_g_name="decoder.ups.1.0.weight_g",
    weight_v_name="decoder.ups.1.0.weight_v",
    bias_name="decoder.ups.1.0.bias",
  ) catch {
    error => raise Reference(error)
  }
}

///|
/// Authenticates `decoder.resblocks.3.activations.0`, the first operation in
/// the three-way AMP stage after released `decoder.ups.1.0`.
pub fn bind_audio_decoder_second_amp_activation(
  metadata : @minimax_h3_spec.MiniMaxH3ModelMetadata,
  config : @minimax_h3_config.MiniMaxH3AudioVaeConfig,
  request : @joint_diffusion_plan.JointDiffusionRequestShape,
) -> @joint_diffusion_vae_reference.JointVaeAudioDecoderAmpActivation raise MiniMaxH3VaeReferenceError {
  let plan = @minimax_h3.build(metadata) catch { error => raise Plan(error) }
  let schema = @minimax_h3_weights.MiniMaxH3AudioVaeSchema::official_v1()
  let prefix = "decoder.resblocks.3.activations.0"
  if config.architecture_schema_version() != 1 ||
    config.decoder_dimension() != 1024 ||
    config.resblock_kernel_sizes() != [3, 7, 11] ||
    config.resblock_dilation_sizes()[0] != [1, 3, 5] ||
    config.resblock_dilation_sizes()[1] != [1, 3, 5] ||
    config.resblock_dilation_sizes()[2] != [1, 3, 5] ||
    !schema.matches_tensor("\{prefix}.act.alpha", Float32, [256L]) ||
    !schema.matches_tensor("\{prefix}.act.beta", Float32, [256L]) ||
    !schema.matches_tensor("\{prefix}.upsample.filter", Float32, [1L, 1L, 12L]) ||
    !schema.matches_tensor("\{prefix}.downsample.lowpass.filter", Float32, [
      1L, 1L, 12L,
    ]) {
    raise InvalidAudioVaeConfig
  }
  let time64 = request.audio_latent_frames().to_int64() * 25L
  let time = time64.to_int()
  if time.to_int64() != time64 {
    raise InvalidAudioVaeConfig
  }
  @joint_diffusion_vae_reference.bind_audio_decoder_amp_activation(
    plan,
    request,
    stage_index=1,
    decoder_batch_count=2,
    channel_count=256,
    time_count=time,
    alpha_name="\{prefix}.act.alpha",
    beta_name="\{prefix}.act.beta",
    upsample_filter_name="\{prefix}.upsample.filter",
    downsample_filter_name="\{prefix}.downsample.lowpass.filter",
  ) catch {
    error => raise Reference(error)
  }
}

///|
/// Authenticates released `decoder.ups.2.0` after the second AMP average.
pub fn bind_audio_decoder_third_upsample(
  metadata : @minimax_h3_spec.MiniMaxH3ModelMetadata,
  config : @minimax_h3_config.MiniMaxH3AudioVaeConfig,
  request : @joint_diffusion_plan.JointDiffusionRequestShape,
) -> @joint_diffusion_vae_reference.JointVaeAudioDecoderUpsampleTransition raise MiniMaxH3VaeReferenceError {
  let plan = @minimax_h3.build(metadata) catch { error => raise Plan(error) }
  let schema = @minimax_h3_weights.MiniMaxH3AudioVaeSchema::official_v1()
  if config.architecture_schema_version() != 1 ||
    config.decoder_dimension() != 1024 ||
    config.decoder_rates().length() != 7 ||
    config.decoder_kernel_sizes().length() != 7 ||
    config.decoder_rates()[2] != 2 ||
    config.decoder_kernel_sizes()[2] != 4 ||
    config.output_channels() != 2 ||
    !schema.matches_tensor("decoder.ups.2.0.weight_g", Float32, [256L, 1L, 1L]) ||
    !schema.matches_tensor("decoder.ups.2.0.weight_v", Float32, [256L, 128L, 4L]) ||
    !schema.matches_tensor("decoder.ups.2.0.bias", Float32, [128L]) {
    raise InvalidAudioVaeConfig
  }
  let input_time64 = request.audio_latent_frames().to_int64() * 25L
  let output_time64 = input_time64 * 2L
  let input_time = input_time64.to_int()
  let output_time = output_time64.to_int()
  if input_time.to_int64() != input_time64 ||
    output_time.to_int64() != output_time64 {
    raise InvalidAudioVaeConfig
  }
  @joint_diffusion_vae_reference.bind_audio_decoder_upsample_transition(
    plan,
    request,
    stage_index=2,
    decoder_batch_count=2,
    input_channels=256,
    output_channels=128,
    input_time~,
    output_time~,
    kernel_size=4,
    stride=2,
    padding=1,
    weight_g_name="decoder.ups.2.0.weight_g",
    weight_v_name="decoder.ups.2.0.weight_v",
    bias_name="decoder.ups.2.0.bias",
  ) catch {
    error => raise Reference(error)
  }
}

///|
/// Authenticates `decoder.resblocks.6.activations.0`, immediately following
/// released `decoder.ups.2.0`.
pub fn bind_audio_decoder_third_amp_activation(
  metadata : @minimax_h3_spec.MiniMaxH3ModelMetadata,
  config : @minimax_h3_config.MiniMaxH3AudioVaeConfig,
  request : @joint_diffusion_plan.JointDiffusionRequestShape,
) -> @joint_diffusion_vae_reference.JointVaeAudioDecoderAmpActivation raise MiniMaxH3VaeReferenceError {
  let plan = @minimax_h3.build(metadata) catch { error => raise Plan(error) }
  let schema = @minimax_h3_weights.MiniMaxH3AudioVaeSchema::official_v1()
  let prefix = "decoder.resblocks.6.activations.0"
  if config.architecture_schema_version() != 1 ||
    config.decoder_dimension() != 1024 ||
    config.resblock_kernel_sizes() != [3, 7, 11] ||
    config.resblock_dilation_sizes()[0] != [1, 3, 5] ||
    config.resblock_dilation_sizes()[1] != [1, 3, 5] ||
    config.resblock_dilation_sizes()[2] != [1, 3, 5] ||
    !schema.matches_tensor("\{prefix}.act.alpha", Float32, [128L]) ||
    !schema.matches_tensor("\{prefix}.act.beta", Float32, [128L]) ||
    !schema.matches_tensor("\{prefix}.upsample.filter", Float32, [1L, 1L, 12L]) ||
    !schema.matches_tensor("\{prefix}.downsample.lowpass.filter", Float32, [
      1L, 1L, 12L,
    ]) {
    raise InvalidAudioVaeConfig
  }
  let time64 = request.audio_latent_frames().to_int64() * 50L
  let time = time64.to_int()
  if time.to_int64() != time64 {
    raise InvalidAudioVaeConfig
  }
  @joint_diffusion_vae_reference.bind_audio_decoder_amp_activation(
    plan,
    request,
    stage_index=2,
    decoder_batch_count=2,
    channel_count=128,
    time_count=time,
    alpha_name="\{prefix}.act.alpha",
    beta_name="\{prefix}.act.beta",
    upsample_filter_name="\{prefix}.upsample.filter",
    downsample_filter_name="\{prefix}.downsample.lowpass.filter",
  ) catch {
    error => raise Reference(error)
  }
}
