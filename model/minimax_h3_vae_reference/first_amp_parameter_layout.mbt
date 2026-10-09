///|
fn amp_align64(value : Int64) -> Int64 {
  (value + 63L) / 64L * 64L
}

///|
fn bind_amp_parameter_layout(
  stage_index : Int,
  channels : Int64,
) -> @joint_audio_amp_parameter_layout.JointAudioAmpParameterLayout raise MiniMaxH3VaeReferenceError {
  if stage_index < 0 || stage_index > 6 || channels <= 0L {
    raise InvalidAudioVaeConfig
  }
  let schema = @minimax_h3_weights.MiniMaxH3AudioVaeSchema::official_v1()
  let inputs : Array[
    @joint_audio_amp_parameter_layout.JointAudioAmpParameterRegionInput,
  ] = []
  let mut offset = 0L
  for block in 0..<3 {
    let checkpoint_block = stage_index * 3 + block
    let kernel : Int64 = match block {
      0 => 3L
      1 => 7L
      _ => 11L
    }
    for activation in 0..<6 {
      let prefix = "decoder.resblocks.\{checkpoint_block}.activations.\{activation}"
      let names = [
        "\{prefix}.act.alpha",
        "\{prefix}.act.beta",
        "\{prefix}.downsample.lowpass.filter",
        "\{prefix}.upsample.filter",
      ]
      let roles : Array[
        @joint_audio_amp_parameter_layout.JointAudioAmpParameterRole,
      ] = [
        ActivationAlpha(block, activation),
        ActivationBeta(block, activation),
        ActivationDownsampleFilter(block, activation),
        ActivationUpsampleFilter(block, activation),
      ]
      for component in 0..<4 {
        let elements = if component < 2 { channels } else { 12L }
        let shape : ReadOnlyArray[Int64] = if component < 2 {
          [channels]
        } else {
          [1L, 1L, 12L]
        }
        if !schema.matches_tensor(names[component], Float32, shape) {
          raise InvalidAudioVaeConfig
        }
        offset = amp_align64(offset)
        inputs.push(
          @joint_audio_amp_parameter_layout.JointAudioAmpParameterRegionInput::new(
            role=roles[component],
            tensor_name=names[component],
            element_count=elements,
            byte_offset=offset,
          ) catch {
            _ => raise InvalidAudioVaeConfig
          },
        )
        offset += elements * 4L
      }
    }
    for pair in 0..<3 {
      for side in 0..<2 {
        let group = if side == 0 { "convs1" } else { "convs2" }
        let prefix = "decoder.resblocks.\{checkpoint_block}.\{group}.\{pair}"
        let names = [
          "\{prefix}.bias",
          "\{prefix}.weight_g",
          "\{prefix}.weight_v",
        ]
        let roles : Array[
          @joint_audio_amp_parameter_layout.JointAudioAmpParameterRole,
        ] = [
          ConvolutionBias(block, pair, side),
          ConvolutionWeightG(block, pair, side),
          ConvolutionWeightV(block, pair, side),
        ]
        let elements = [channels, channels, channels * channels * kernel]
        let shapes : Array[ReadOnlyArray[Int64]] = [
          [channels],
          [channels, 1L, 1L],
          [channels, channels, kernel],
        ]
        for component in 0..<3 {
          if !schema.matches_tensor(
              names[component],
              Float32,
              shapes[component],
            ) {
            raise InvalidAudioVaeConfig
          }
          offset = amp_align64(offset)
          inputs.push(
            @joint_audio_amp_parameter_layout.JointAudioAmpParameterRegionInput::new(
              role=roles[component],
              tensor_name=names[component],
              element_count=elements[component],
              byte_offset=offset,
            ) catch {
              _ => raise InvalidAudioVaeConfig
            },
          )
          offset += elements[component] * 4L
        }
      }
    }
  }
  @joint_audio_amp_parameter_layout.JointAudioAmpParameterLayout::new(inputs) catch {
    _ => raise InvalidAudioVaeConfig
  }
}

///|
pub fn bind_first_amp_parameter_layout() -> @joint_audio_amp_parameter_layout.JointAudioAmpParameterLayout raise MiniMaxH3VaeReferenceError {
  bind_audio_decoder_amp_parameter_layout(stage_index=0)
}

///|
pub fn bind_second_amp_parameter_layout() -> @joint_audio_amp_parameter_layout.JointAudioAmpParameterLayout raise MiniMaxH3VaeReferenceError {
  bind_audio_decoder_amp_parameter_layout(stage_index=1)
}

///|
/// All seven released AMP stages share one parameter-layout derivation.
pub fn bind_audio_decoder_amp_parameter_layout(
  stage_index~ : Int,
) -> @joint_audio_amp_parameter_layout.JointAudioAmpParameterLayout raise MiniMaxH3VaeReferenceError {
  guard stage_index >= 0 && stage_index < 7 else { raise InvalidAudioVaeConfig }
  bind_amp_parameter_layout(stage_index, 1024L >> (stage_index + 1))
}
