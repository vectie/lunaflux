///|
fn valid_tensor_name(name : StringView) -> Bool {
  !name.is_empty() &&
  name.length() <= 256 &&
  name.all(c => c is ('a'..='z' | 'A'..='Z' | '0'..='9' | '_' | '.'))
}

///|
pub fn bind_audio_decoder_input_projection(
  plan : @joint_diffusion_plan.JointDiffusionPlan,
  request : @joint_diffusion_plan.JointDiffusionRequestShape,
  decoder_batch_count~ : Int,
  weight_name~ : String,
  bias_name~ : String,
) -> JointVaeAudioDecoderInputProjection raise JointVaeDecodeReferenceError {
  if !request_matches_plan(plan, request) || decoder_batch_count <= 0 {
    raise InvalidRequest
  }
  if !valid_tensor_name(weight_name) || !valid_tensor_name(bias_name) {
    raise InvalidProjectionTensorIdentity
  }
  let shape = plan.shapes()
  let input_channels = shape.audio_latent_channels()
  let output_channels = shape.audio_vae_latent_dimension()
  let time = request.audio_latent_frames().to_int64()
  let input_elements = checked_product([
    decoder_batch_count.to_int64(),
    input_channels.to_int64(),
    time,
  ])
  let weight_elements = checked_product([
    output_channels.to_int64(),
    input_channels.to_int64(),
    1L,
  ])
  let output_elements = checked_product([
    decoder_batch_count.to_int64(),
    output_channels.to_int64(),
    time,
  ])
  let canonical = StringBuilder()
  canonical <+ "lunaflux.joint-vae-audio-decoder-input-projection.v1"
  canonical <+ "|plan=\{plan.plan_digest().as_hex()}"
  canonical <+ "|layout=b-c-t|storage=f32-input-weight-bias-output"
  canonical <+
    "|batch=\{decoder_batch_count}|time=\{request.audio_latent_frames()}"
  canonical <+ "|input=\{input_channels}|output=\{output_channels}|kernel=1"
  canonical <+ "|weight=\{weight_name}|bias=\{bias_name}"
  {
    plan_digest: plan.plan_digest(),
    request_shape: request,
    decoder_batch_count,
    input_channels,
    output_channels,
    kernel_size: 1,
    storage: Float32InputWeightBiasOutput,
    layout: BatchChannelTime,
    weight_name,
    bias_name,
    input_element_count: input_elements,
    weight_element_count: weight_elements,
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
/// Scalar oracle for F32 `[B,C,T]` pointwise projection. Accumulation starts
/// from bias and visits input channels in ascending order.
pub fn project_audio_decoder_input(
  contract : JointVaeAudioDecoderInputProjection,
  input : ArrayView[Float],
  weight : ArrayView[Float],
  bias : ArrayView[Float],
) -> ReadOnlyArray[Float] raise JointVaeDecodeReferenceError {
  if input.length().to_int64() != contract.input_element_count ||
    weight.length().to_int64() != contract.weight_element_count ||
    bias.length().to_int64() != contract.bias_element_count {
    raise InvalidProjectionInput
  }
  let output : Array[Float] = []
  let time = contract.request_shape.audio_latent_frames()
  for batch in 0..<contract.decoder_batch_count {
    for output_channel in 0..<contract.output_channels {
      for time_index in 0..<time {
        let mut sum = bias[output_channel]
        if !finite_f32(sum) {
          raise InvalidProjectionInput
        }
        for input_channel in 0..<contract.input_channels {
          let value = input[(batch * contract.input_channels + input_channel) *
            time +
            time_index]
          let coefficient = weight[output_channel * contract.input_channels +
            input_channel]
          if !finite_f32(value) || !finite_f32(coefficient) {
            raise InvalidProjectionInput
          }
          sum = sum + value * coefficient
        }
        output.push(sum)
      }
    }
  }
  ReadOnlyArray::from_array(output)
}
