///|
pub fn bind_audio_decoder_conv_pre(
  plan : @joint_diffusion_plan.JointDiffusionPlan,
  request : @joint_diffusion_plan.JointDiffusionRequestShape,
  decoder_batch_count~ : Int,
  weight_g_name~ : String,
  weight_v_name~ : String,
  bias_name~ : String,
) -> JointVaeAudioDecoderConvPre raise JointVaeDecodeReferenceError {
  if !request_matches_plan(plan, request) || decoder_batch_count <= 0 {
    raise InvalidRequest
  }
  if !valid_tensor_name(weight_g_name) ||
    !valid_tensor_name(weight_v_name) ||
    !valid_tensor_name(bias_name) {
    raise InvalidProjectionTensorIdentity
  }
  let shape = plan.shapes()
  let input_channels = shape.audio_vae_latent_dimension()
  let output_channels = shape.audio_vae_decoder_dimension()
  let kernel_size = 7
  let padding = 3
  let stride = 1
  let time = request.audio_latent_frames().to_int64()
  let input_elements = checked_product([
    decoder_batch_count.to_int64(),
    input_channels.to_int64(),
    time,
  ])
  let weight_v_elements = checked_product([
    output_channels.to_int64(),
    input_channels.to_int64(),
    kernel_size.to_int64(),
  ])
  let output_elements = checked_product([
    decoder_batch_count.to_int64(),
    output_channels.to_int64(),
    time,
  ])
  let canonical = StringBuilder()
  canonical <+ "lunaflux.joint-vae-audio-decoder-conv-pre.v1"
  canonical <+ "|plan=\{plan.plan_digest().as_hex()}|layout=b-c-t"
  canonical <+ "|storage=f32-input-g-v-bias-normalized-weight-output"
  canonical <+ "|axes=per-output-input-kernel|batch=\{decoder_batch_count}"
  canonical <+ "|input=\{input_channels}|output=\{output_channels}"
  canonical <+
    "|time=\{request.audio_latent_frames()}|kernel=7|padding=3|stride=1"
  canonical <+ "|g=\{weight_g_name}|v=\{weight_v_name}|bias=\{bias_name}"
  {
    plan_digest: plan.plan_digest(),
    request_shape: request,
    decoder_batch_count,
    input_channels,
    output_channels,
    kernel_size,
    padding,
    stride,
    storage: Float32InputWeightGWeightVBiasWorkspaceOutput,
    axes: PerOutputChannelInputAndKernel,
    layout: BatchChannelTime,
    weight_g_name,
    weight_v_name,
    bias_name,
    input_element_count: input_elements,
    weight_g_element_count: output_channels.to_int64(),
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
/// Scalar oracle for the two ordered stages. Every `weight_v` output-channel
/// slice must contain finite values with a finite, nonzero F32 sum of squares;
/// this is a runtime data precondition and is not inferred from tensor metadata.
pub fn execute_audio_decoder_conv_pre(
  contract : JointVaeAudioDecoderConvPre,
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
  let per_output = contract.input_channels * contract.kernel_size
  for output_channel in 0..<contract.output_channels {
    let mut sum_squares = 0.0F
    let base = output_channel * per_output
    for input_channel in 0..<contract.input_channels {
      for kernel_index in 0..<contract.kernel_size {
        let index = base + input_channel * contract.kernel_size + kernel_index
        let value = weight_v[index]
        if !finite_f32(value) {
          raise InvalidProjectionInput
        }
        sum_squares = sum_squares + value * value
      }
    }
    let magnitude = weight_g[output_channel]
    if !finite_f32(magnitude) || !finite_f32(sum_squares) || sum_squares <= 0.0F {
      raise InvalidProjectionInput
    }
    let scale = magnitude / sum_squares.sqrt()
    for index in base..<(base + per_output) {
      let value = weight_v[index] * scale
      if !finite_f32(value) {
        raise InvalidProjectionInput
      }
      normalized[index] = value
    }
  }
  let time = contract.request_shape.audio_latent_frames()
  let output : Array[Float] = []
  for batch in 0..<contract.decoder_batch_count {
    for output_channel in 0..<contract.output_channels {
      for output_time in 0..<time {
        let mut sum = bias[output_channel]
        if !finite_f32(sum) {
          raise InvalidProjectionInput
        }
        let weight_base = output_channel * per_output
        for input_channel in 0..<contract.input_channels {
          for kernel_index in 0..<contract.kernel_size {
            let input_time = output_time + kernel_index - contract.padding
            if input_time >= 0 && input_time < time {
              let input_index = (
                  batch * contract.input_channels + input_channel
                ) *
                time +
                input_time
              let weight_index = weight_base +
                input_channel * contract.kernel_size +
                kernel_index
              let value = input[input_index]
              if !finite_f32(value) {
                raise InvalidProjectionInput
              }
              sum = sum + value * normalized[weight_index]
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
