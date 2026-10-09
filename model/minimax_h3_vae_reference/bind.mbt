///|
pub(all) suberror MiniMaxH3VaeReferenceError {
  Plan(@joint_diffusion_plan.JointDiffusionPlanError)
  InvalidVideoVaeConfig
  InvalidAudioVaeConfig
  Reference(@joint_diffusion_vae_reference.JointVaeDecodeReferenceError)
} derive(Debug, Eq)

///|
/// Thin family binding from authenticated MiniMax config values to the
/// family-neutral request-owned inverse-normalization contract.
pub fn bind_video_latent_destandardization(
  metadata : @minimax_h3_spec.MiniMaxH3ModelMetadata,
  config : @minimax_h3_config.MiniMaxH3VideoVaeConfig,
  request : @joint_diffusion_plan.JointDiffusionRequestShape,
) -> @joint_diffusion_vae_reference.JointVaeLatentDestandardization raise MiniMaxH3VaeReferenceError {
  let plan = @minimax_h3.build(metadata) catch { error => raise Plan(error) }
  if config.architecture_schema_version() != 1 ||
    config.latent_channels() != plan.shapes().video_latent_channels() ||
    config.latent_means().length() != config.latent_channels() ||
    config.latent_standard_deviations().length() != config.latent_channels() {
    raise InvalidVideoVaeConfig
  }
  @joint_diffusion_vae_reference.bind_video_latent_destandardization(
    plan,
    request,
    means=config.latent_means(),
    standard_deviations=config.latent_standard_deviations(),
  ) catch {
    error => raise Reference(error)
  }
}

///|
/// Binds the official stereo `[2,32,T]` F32 inverse-normalization prefix.
pub fn bind_audio_latent_destandardization(
  metadata : @minimax_h3_spec.MiniMaxH3ModelMetadata,
  config : @minimax_h3_config.MiniMaxH3AudioVaeConfig,
  request : @joint_diffusion_plan.JointDiffusionRequestShape,
) -> @joint_diffusion_vae_reference.JointVaeAudioLatentDestandardization raise MiniMaxH3VaeReferenceError {
  let plan = @minimax_h3.build(metadata) catch { error => raise Plan(error) }
  if config.architecture_schema_version() != 1 ||
    config.latent_channels() != plan.shapes().audio_latent_channels() ||
    config.output_channels() != 2 ||
    config.latent_means().length() != config.latent_channels() ||
    config.latent_standard_deviations().length() != config.latent_channels() {
    raise InvalidAudioVaeConfig
  }
  @joint_diffusion_vae_reference.bind_audio_latent_destandardization(
    plan,
    request,
    batch_count=config.output_channels(),
    means=config.latent_means(),
    standard_deviations=config.latent_standard_deviations(),
  ) catch {
    error => raise Reference(error)
  }
}

///|
/// Authenticates the released F32 `dec_in_proj` tensors and binds the exact
/// channel-major pointwise projection that begins AudioVAE decoding.
pub fn bind_audio_decoder_input_projection(
  metadata : @minimax_h3_spec.MiniMaxH3ModelMetadata,
  config : @minimax_h3_config.MiniMaxH3AudioVaeConfig,
  request : @joint_diffusion_plan.JointDiffusionRequestShape,
) -> @joint_diffusion_vae_reference.JointVaeAudioDecoderInputProjection raise MiniMaxH3VaeReferenceError {
  let plan = @minimax_h3.build(metadata) catch { error => raise Plan(error) }
  let schema = @minimax_h3_weights.MiniMaxH3AudioVaeSchema::official_v1()
  if config.architecture_schema_version() != 1 ||
    config.latent_channels() != plan.shapes().audio_latent_channels() ||
    config.latent_dimension() != plan.shapes().audio_vae_latent_dimension() ||
    config.decoder_dimension() != plan.shapes().audio_vae_decoder_dimension() ||
    config.output_channels() != 2 ||
    !schema.matches_tensor("dec_in_proj.weight", Float32, [2048L, 32L, 1L]) ||
    !schema.matches_tensor("dec_in_proj.bias", Float32, [2048L]) {
    raise InvalidAudioVaeConfig
  }
  @joint_diffusion_vae_reference.bind_audio_decoder_input_projection(
    plan,
    request,
    decoder_batch_count=config.output_channels(),
    weight_name="dec_in_proj.weight",
    bias_name="dec_in_proj.bias",
  ) catch {
    error => raise Reference(error)
  }
}

///|
/// Authenticates the released F32 weight-normalized BigVGAN `conv_pre`.
pub fn bind_audio_decoder_conv_pre(
  metadata : @minimax_h3_spec.MiniMaxH3ModelMetadata,
  config : @minimax_h3_config.MiniMaxH3AudioVaeConfig,
  request : @joint_diffusion_plan.JointDiffusionRequestShape,
) -> @joint_diffusion_vae_reference.JointVaeAudioDecoderConvPre raise MiniMaxH3VaeReferenceError {
  let plan = @minimax_h3.build(metadata) catch { error => raise Plan(error) }
  let schema = @minimax_h3_weights.MiniMaxH3AudioVaeSchema::official_v1()
  if config.architecture_schema_version() != 1 ||
    config.latent_dimension() != plan.shapes().audio_vae_latent_dimension() ||
    config.decoder_dimension() != plan.shapes().audio_vae_decoder_dimension() ||
    config.output_channels() != 2 ||
    !schema.matches_tensor("decoder.conv_pre.weight_g", Float32, [1024L, 1L, 1L]) ||
    !schema.matches_tensor("decoder.conv_pre.weight_v", Float32, [
      1024L, 2048L, 7L,
    ]) ||
    !schema.matches_tensor("decoder.conv_pre.bias", Float32, [1024L]) {
    raise InvalidAudioVaeConfig
  }
  @joint_diffusion_vae_reference.bind_audio_decoder_conv_pre(
    plan,
    request,
    decoder_batch_count=config.output_channels(),
    weight_g_name="decoder.conv_pre.weight_g",
    weight_v_name="decoder.conv_pre.weight_v",
    bias_name="decoder.conv_pre.bias",
  ) catch {
    error => raise Reference(error)
  }
}

///|
/// Authenticates the first released F32 BigVGAN transposed-convolution stage.
pub fn bind_audio_decoder_first_upsample(
  metadata : @minimax_h3_spec.MiniMaxH3ModelMetadata,
  config : @minimax_h3_config.MiniMaxH3AudioVaeConfig,
  request : @joint_diffusion_plan.JointDiffusionRequestShape,
) -> @joint_diffusion_vae_reference.JointVaeAudioDecoderFirstUpsample raise MiniMaxH3VaeReferenceError {
  let plan = @minimax_h3.build(metadata) catch { error => raise Plan(error) }
  let schema = @minimax_h3_weights.MiniMaxH3AudioVaeSchema::official_v1()
  if config.architecture_schema_version() != 1 ||
    config.decoder_dimension() != plan.shapes().audio_vae_decoder_dimension() ||
    config.decoder_rates()[0] != 5 ||
    config.decoder_kernel_sizes()[0] != 9 ||
    config.output_channels() != 2 ||
    !schema.matches_tensor("decoder.ups.0.0.weight_g", Float32, [1024L, 1L, 1L]) ||
    !schema.matches_tensor("decoder.ups.0.0.weight_v", Float32, [
      1024L, 512L, 9L,
    ]) ||
    !schema.matches_tensor("decoder.ups.0.0.bias", Float32, [512L]) {
    raise InvalidAudioVaeConfig
  }
  @joint_diffusion_vae_reference.bind_audio_decoder_first_upsample(
    plan,
    request,
    decoder_batch_count=config.output_channels(),
    weight_g_name="decoder.ups.0.0.weight_g",
    weight_v_name="decoder.ups.0.0.weight_v",
    bias_name="decoder.ups.0.0.bias",
  ) catch {
    error => raise Reference(error)
  }
}

///|
pub fn bind_audio_decoder_first_amp_activation(
  metadata : @minimax_h3_spec.MiniMaxH3ModelMetadata,
  config : @minimax_h3_config.MiniMaxH3AudioVaeConfig,
  request : @joint_diffusion_plan.JointDiffusionRequestShape,
) -> @joint_diffusion_vae_reference.JointVaeAudioDecoderFirstAmpActivation raise MiniMaxH3VaeReferenceError {
  let plan = @minimax_h3.build(metadata) catch { error => raise Plan(error) }
  let schema = @minimax_h3_weights.MiniMaxH3AudioVaeSchema::official_v1()
  if config.architecture_schema_version() != 1 ||
    config.decoder_dimension() != 1024 ||
    config.resblock_kernel_sizes()[0] != 3 ||
    config.resblock_dilation_sizes()[0] != [1, 3, 5] ||
    !schema.matches_tensor(
      "decoder.resblocks.0.activations.0.act.alpha",
      Float32,
      [512L],
    ) ||
    !schema.matches_tensor(
      "decoder.resblocks.0.activations.0.act.beta",
      Float32,
      [512L],
    ) ||
    !schema.matches_tensor(
      "decoder.resblocks.0.activations.0.upsample.filter",
      Float32,
      [1L, 1L, 12L],
    ) ||
    !schema.matches_tensor(
      "decoder.resblocks.0.activations.0.downsample.lowpass.filter",
      Float32,
      [1L, 1L, 12L],
    ) {
    raise InvalidAudioVaeConfig
  }
  @joint_diffusion_vae_reference.bind_audio_decoder_first_amp_activation(
    plan,
    request,
    decoder_batch_count=config.output_channels(),
    alpha_name="decoder.resblocks.0.activations.0.act.alpha",
    beta_name="decoder.resblocks.0.activations.0.act.beta",
    upsample_filter_name="decoder.resblocks.0.activations.0.upsample.filter",
    downsample_filter_name="decoder.resblocks.0.activations.0.downsample.lowpass.filter",
  ) catch {
    error => raise Reference(error)
  }
}
