///|
pub fn bind_audio_decoder_first_upsample(
  plan : @joint_diffusion_plan.JointDiffusionPlan,
  request : @joint_diffusion_plan.JointDiffusionRequestShape,
  decoder_batch_count~ : Int,
  weight_g_name~ : String,
  weight_v_name~ : String,
  bias_name~ : String,
) -> JointVaeAudioDecoderFirstUpsample raise JointVaeDecodeReferenceError {
  if !request_matches_plan(plan, request) || decoder_batch_count <= 0 {
    raise InvalidRequest
  }
  if !valid_tensor_name(weight_g_name) ||
    !valid_tensor_name(weight_v_name) ||
    !valid_tensor_name(bias_name) {
    raise InvalidProjectionTensorIdentity
  }
  let input_channels = plan.shapes().audio_vae_decoder_dimension()
  let output_channels = input_channels / 2
  let input_time = request.audio_latent_frames()
  let kernel_size = 9
  let stride = 5
  let padding = 2
  let output_time = (input_time - 1) * stride - 2 * padding + kernel_size
  let input_elements = checked_product([
    decoder_batch_count.to_int64(),
    input_channels.to_int64(),
    input_time.to_int64(),
  ])
  let weight_v_elements = checked_product([
    input_channels.to_int64(),
    output_channels.to_int64(),
    kernel_size.to_int64(),
  ])
  let output_elements = checked_product([
    decoder_batch_count.to_int64(),
    output_channels.to_int64(),
    output_time.to_int64(),
  ])
  let canonical = StringBuilder()
  canonical <+ "lunaflux.joint-vae-audio-decoder-first-upsample.v1"
  canonical <+ "|plan=\{plan.plan_digest().as_hex()}|layout=b-c-t"
  canonical <+ "|storage=f32-input-g-v-bias-normalized-weight-output"
  canonical <+ "|axes=per-input-output-kernel|batch=\{decoder_batch_count}"
  canonical <+ "|input=\{input_channels}|output=\{output_channels}"
  canonical <+ "|time=\{input_time}:\{output_time}|kernel=9|stride=5|padding=2"
  canonical <+ "|g=\{weight_g_name}|v=\{weight_v_name}|bias=\{bias_name}"
  {
    plan_digest: plan.plan_digest(),
    request_shape: request,
    decoder_batch_count,
    input_channels,
    output_channels,
    input_time,
    output_time,
    kernel_size,
    stride,
    padding,
    storage: Float32InputWeightGWeightVBiasWorkspaceOutput,
    axes: PerInputChannelOutputAndKernel,
    layout: BatchChannelTime,
    weight_g_name,
    weight_v_name,
    bias_name,
    input_element_count: input_elements,
    weight_g_element_count: input_channels.to_int64(),
    weight_v_element_count: weight_v_elements,
    bias_element_count: output_channels.to_int64(),
    output_element_count: output_elements,
    digest: {
      hex: @crypto.bytes_to_hex_string(
        @crypto.sha256(@utf8.encode(canonical.to_string())),
      ),
    },
  }
}

///|
/// Scalar oracle. Each input-channel `weight_v` slice must have a finite,
/// nonzero F32 norm. This runtime value precondition is not metadata evidence.
pub fn execute_audio_decoder_first_upsample(
  contract : JointVaeAudioDecoderFirstUpsample,
  input : ArrayView[Float],
  weight_g : ArrayView[Float],
  weight_v : ArrayView[Float],
  bias : ArrayView[Float],
) -> ReadOnlyArray[Float] raise JointVaeDecodeReferenceError {
  if input.length().to_int64() != contract.input_element_count ||
    weight_g.length().to_int64() != contract.weight_g_element_count ||
    weight_v.length().to_int64() != contract.weight_v_element_count ||
    bias.length().to_int64() != contract.bias_element_count {
    raise InvalidProjectionInput
  }
  let normalized = weight_v.to_owned()
  let per_input = contract.output_channels * contract.kernel_size
  for input_channel in 0..<contract.input_channels {
    let base = input_channel * per_input
    let mut sum_squares = 0.0F
    for output_channel in 0..<contract.output_channels {
      for kernel_index in 0..<contract.kernel_size {
        let value = weight_v[base +
          output_channel * contract.kernel_size +
          kernel_index]
        if !finite_f32(value) {
          raise InvalidProjectionInput
        }
        sum_squares = sum_squares + value * value
      }
    }
    let magnitude = weight_g[input_channel]
    if !finite_f32(magnitude) || !finite_f32(sum_squares) || sum_squares <= 0.0F {
      raise InvalidProjectionInput
    }
    let scale = magnitude / sum_squares.sqrt()
    for index in base..<(base + per_input) {
      let value = weight_v[index] * scale
      if !finite_f32(value) {
        raise InvalidProjectionInput
      }
      normalized[index] = value
    }
  }
  let output : Array[Float] = []
  for batch in 0..<contract.decoder_batch_count {
    for output_channel in 0..<contract.output_channels {
      for output_time in 0..<contract.output_time {
        let mut sum = bias[output_channel]
        for input_channel in 0..<contract.input_channels {
          let weight_base = input_channel * per_input +
            output_channel * contract.kernel_size
          for input_time in 0..<contract.input_time {
            let kernel_index = output_time -
              input_time * contract.stride +
              contract.padding
            if kernel_index >= 0 && kernel_index < contract.kernel_size {
              let input_index = (
                  batch * contract.input_channels + input_channel
                ) *
                contract.input_time +
                input_time
              sum = sum +
                input[input_index] * normalized[weight_base + kernel_index]
            }
          }
        }
        if !finite_f32(sum) {
          raise InvalidProjectionInput
        }
        output.push(sum)
      }
    }
  }
  ReadOnlyArray::from_array(output)
}
