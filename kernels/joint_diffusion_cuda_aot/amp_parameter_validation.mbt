///|
fn validate_amp_parameter_geometry(
  layout : @joint_audio_amp_parameter_layout.JointAudioAmpParameterLayout,
  channels : Int,
) -> Unit raise JointDiffusionCudaAotError {
  if layout.regions().length() != 126 || channels <= 0 {
    raise Invalid(Requirement)
  }
  for block in 0..<3 {
    let kernel = [3L, 7L, 11L][block]
    for activation in 0..<6 {
      for
        role in [
          @joint_audio_amp_parameter_layout.ActivationAlpha(block, activation),
          ActivationBeta(block, activation),
        ] {
        let region = layout.regions()[amp_parameter_index(layout, role)]
        if region.element_count() != channels.to_int64() {
          raise Invalid(Requirement)
        }
      }
      for
        role in [
          @joint_audio_amp_parameter_layout.ActivationDownsampleFilter(
            block, activation,
          ),
          ActivationUpsampleFilter(block, activation),
        ] {
        let region = layout.regions()[amp_parameter_index(layout, role)]
        if region.element_count() != 12L {
          raise Invalid(Requirement)
        }
      }
    }
    for pair in 0..<3 {
      for side in 0..<2 {
        for
          role in [
            @joint_audio_amp_parameter_layout.ConvolutionBias(block, pair, side),
            ConvolutionWeightG(block, pair, side),
          ] {
          let region = layout.regions()[amp_parameter_index(layout, role)]
          if region.element_count() != channels.to_int64() {
            raise Invalid(Requirement)
          }
        }
        let weight = layout.regions()[amp_parameter_index(
            layout,
            ConvolutionWeightV(block, pair, side),
          )]
        let expected = checked_mul(
          checked_mul(channels.to_int64(), channels.to_int64()),
          kernel,
        )
        if weight.element_count() != expected {
          raise Invalid(Requirement)
        }
      }
    }
  }
}
