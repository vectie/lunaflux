///|
/// Compile the complete seven-stage upsampling spine. AMP residual bodies
/// execute between transitions; this is not a claim of a complete decoder.
pub fn bind_audio_decoder_upsample_stages(
  metadata : @minimax_h3_spec.MiniMaxH3ModelMetadata,
  config : @minimax_h3_config.MiniMaxH3AudioVaeConfig,
  request : @joint_diffusion_plan.JointDiffusionRequestShape,
) -> ReadOnlyArray[
  @joint_diffusion_vae_reference.JointVaeAudioDecoderUpsampleTransition,
] raise MiniMaxH3VaeReferenceError {
  let model = @minimax_h3.build(metadata) catch { error => raise Plan(error) }
  guard config.architecture_schema_version() == 1 &&
    config.decoder_dimension() == 1024 &&
    config.decoder_rates() == [5, 5, 2, 2, 2, 2, 2] &&
    config.decoder_kernel_sizes() == [9, 9, 4, 4, 4, 4, 4] &&
    config.output_channels() == 2 else {
    raise InvalidAudioVaeConfig
  }
  let schema = @minimax_h3_weights.MiniMaxH3AudioVaeSchema::official_v1()
  let stages = []
  for
    stage in 0..<config.decoder_rates().length()
    time = request.audio_latent_frames() {
    let stride = config.decoder_rates()[stage]
    let kernel = config.decoder_kernel_sizes()[stage]
    let input = config.decoder_dimension() >> stage
    let output = input / 2
    let next_time = time.to_int64() * stride.to_int64()
    guard next_time <= 2147483647L else { raise InvalidAudioVaeConfig }
    let prefix = "decoder.ups.\{stage}.0"
    guard schema.matches_tensor(prefix + ".weight_g", Float32, [
        input.to_int64(),
        1L,
        1L,
      ]) &&
      schema.matches_tensor(prefix + ".weight_v", Float32, [
        input.to_int64(),
        output.to_int64(),
        kernel.to_int64(),
      ]) &&
      schema.matches_tensor(prefix + ".bias", Float32, [output.to_int64()]) else {
      raise InvalidAudioVaeConfig
    }
    stages.push(
      @joint_diffusion_vae_reference.bind_audio_decoder_upsample_transition(
        model,
        request,
        stage_index=stage,
        decoder_batch_count=model.shapes().audio_latent_streams(),
        input_channels=input,
        output_channels=output,
        input_time=time,
        output_time=next_time.to_int(),
        kernel_size=kernel,
        stride~,
        padding=(kernel - stride) / 2,
        weight_g_name=prefix + ".weight_g",
        weight_v_name=prefix + ".weight_v",
        bias_name=prefix + ".bias",
      ) catch {
        error => raise Reference(error)
      },
    )
    continue next_time.to_int()
  }
  ReadOnlyArray::from_array(stages)
}
