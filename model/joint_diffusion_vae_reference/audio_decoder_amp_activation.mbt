///|
pub fn bind_audio_decoder_amp_activation(
  plan : @joint_diffusion_plan.JointDiffusionPlan,
  request : @joint_diffusion_plan.JointDiffusionRequestShape,
  stage_index~ : Int,
  decoder_batch_count~ : Int,
  channel_count~ : Int,
  time_count~ : Int,
  alpha_name~ : String,
  beta_name~ : String,
  upsample_filter_name~ : String,
  downsample_filter_name~ : String,
) -> JointVaeAudioDecoderAmpActivation raise JointVaeDecodeReferenceError {
  if !request_matches_plan(plan, request) ||
    stage_index < 0 ||
    decoder_batch_count <= 0 ||
    channel_count <= 0 ||
    time_count <= 0 ||
    !valid_tensor_name(alpha_name) ||
    !valid_tensor_name(beta_name) ||
    !valid_tensor_name(upsample_filter_name) ||
    !valid_tensor_name(downsample_filter_name) ||
    upsample_filter_name == downsample_filter_name {
    raise InvalidProjectionTensorIdentity
  }
  let upsample_time64 = checked_product([time_count.to_int64(), 2L])
  let upsample_time = upsample_time64.to_int()
  if upsample_time.to_int64() != upsample_time64 {
    raise GeometryOverflow
  }
  let elements = checked_product([
    decoder_batch_count.to_int64(),
    channel_count.to_int64(),
    time_count.to_int64(),
  ])
  let intermediate = checked_product([
    decoder_batch_count.to_int64(),
    channel_count.to_int64(),
    upsample_time64,
  ])
  let canonical = StringBuilder()
  canonical <+ "lunaflux.joint-vae-audio-decoder-amp-activation.v1"
  canonical <+
    "|plan=\{plan.plan_digest().as_hex()}|stage=\{stage_index}|layout=b-c-t|storage=f32-input-alpha-beta-up-filter-down-filter-workspaces-output"
  canonical <+
    "|batch=\{decoder_batch_count}|channels=\{channel_count}|time=\{time_count}:\{upsample_time}|ratio=2|kernel=12"
  canonical <+
    "|alpha=\{alpha_name}|beta=\{beta_name}|up=\{upsample_filter_name}|down=\{downsample_filter_name}"
  {
    plan_digest: plan.plan_digest(),
    request_shape: request,
    stage_index,
    decoder_batch_count,
    channel_count,
    time_count,
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
fn execute_audio_decoder_amp_activation_values(
  channel_count~ : Int,
  time_count~ : Int,
  upsample_time~ : Int,
  element_count~ : Int64,
  intermediate_element_count~ : Int64,
  input : ArrayView[Float],
  alpha : ArrayView[Float],
  beta : ArrayView[Float],
  upsample_filter : ArrayView[Float],
  downsample_filter : ArrayView[Float],
) -> ReadOnlyArray[Float] raise JointVaeDecodeReferenceError {
  if input.length().to_int64() != element_count ||
    alpha.length() != channel_count ||
    beta.length() != channel_count ||
    upsample_filter.length() != 12 ||
    downsample_filter.length() != 12 {
    raise InvalidProjectionInput
  }
  let upsampled = Array::make(intermediate_element_count.to_int(), 0.0F)
  let activated = Array::make(intermediate_element_count.to_int(), 0.0F)
  let output = Array::make(element_count.to_int(), 0.0F)
  for flat in 0..<upsampled.length() {
    let t = flat % upsample_time
    let outer = flat / upsample_time
    let full_t = t + 15
    let mut sum = 0.0F
    for kernel in 0..<12 {
      let numerator = full_t - kernel
      if numerator >= 0 && numerator % 2 == 0 {
        let padded = numerator / 2
        if padded < time_count + 10 {
          let source_t = (padded - 5).clamp(min=0, max=time_count - 1)
          sum += input[outer * time_count + source_t] * upsample_filter[kernel]
        }
      }
    }
    upsampled[flat] = 2.0F * sum
  }
  for flat, value in upsampled {
    let channel = flat / upsample_time % channel_count
    let a = @math.expf(alpha[channel])
    let b = @math.expf(beta[channel])
    let sine = @math.sinf(a * value)
    activated[flat] = value + sine * sine / (b + 1.0e-9F)
  }
  for flat in 0..<output.length() {
    let t = flat % time_count
    let outer = flat / time_count
    let mut sum = 0.0F
    for kernel in 0..<12 {
      let source_t = (t * 2 + kernel - 5).clamp(min=0, max=upsample_time - 1)
      sum += activated[outer * upsample_time + source_t] *
        downsample_filter[kernel]
    }
    output[flat] = sum
  }
  ReadOnlyArray::from_array(output)
}

///|
pub fn execute_audio_decoder_amp_activation(
  contract : JointVaeAudioDecoderAmpActivation,
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
