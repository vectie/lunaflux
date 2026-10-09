///|
fn admission_test_upsample_transition_roles(
  stage : Int,
) -> Array[JointDiffusionLaunchOperandRole] {
  if stage == 1 {
    [
      AudioDecoderFirstAmpStageOutput,
      AudioDecoderUpsampleTransitionWeightG,
      AudioDecoderUpsampleTransitionWeightV,
      AudioDecoderUpsampleTransitionBias,
      AudioDecoderUpsampleTransitionNormalizedWeightWorkspace,
      AudioDecoderUpsampleTransitionOutput,
    ]
  } else {
    [
      AudioDecoderAmpStageOutput,
      AudioDecoderUpsampleTransition2WeightG,
      AudioDecoderUpsampleTransition2WeightV,
      AudioDecoderUpsampleTransition2Bias,
      AudioDecoderUpsampleTransition2NormalizedWeightWorkspace,
      AudioDecoderUpsampleTransition2Output,
    ]
  }
}

///|
fn admission_test_amp_activation_roles(
  stage : Int,
) -> Array[JointDiffusionLaunchOperandRole] {
  let input : JointDiffusionLaunchOperandRole = if stage == 1 {
    AudioDecoderUpsampleTransitionOutput
  } else {
    AudioDecoderUpsampleTransition2Output
  }
  let output : JointDiffusionLaunchOperandRole = if stage == 1 {
    AudioDecoderAmpActivationOutput
  } else {
    AudioDecoderAmpActivation2Output
  }
  [
    input,
    AudioDecoderAmpAlpha,
    AudioDecoderAmpBeta,
    AudioDecoderAmpUpsampleFilter,
    AudioDecoderAmpDownsampleFilter,
    AudioDecoderAmpUpsampleWorkspace,
    AudioDecoderAmpActivatedWorkspace,
    output,
  ]
}
