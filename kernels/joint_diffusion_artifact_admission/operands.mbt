///|
fn expected_operand_roles(
  operation : @joint_diffusion_kernel_requirements.JointDiffusionSemanticOperation,
) -> ReadOnlyArray[JointDiffusionLaunchOperandRole] {
  match operation {
    TokenTextConditioning(_) => [TokenIdsInput, TextConditioningOutput]
    OrderedKeyframeConditioning(_, _) =>
      [KeyframeMediaInput, OrderedConditioningOutput]
    OrderedReferenceConditioning(_, _, _, _, _) =>
      [OrderedReferenceMediaInput, OrderedConditioningOutput]
    VideoLatentProjection(_, _, _, _, _) =>
      [VideoLatentInput, ProjectionWeight, ProjectionBias, JointHiddenOutput]
    AudioLatentProjection(_, _) =>
      [AudioLatentInput, ProjectionWeight, ProjectionBias, JointHiddenOutput]
    JointTransformerAttention(_, _, _, _, _) =>
      [JointHiddenInput, ModelWeights, JointHiddenOutput, Workspace]
    JointTransformerMlp(_, _, _, _) =>
      [
        JointHiddenInput,
        TransformerMlpPackedGateUpWeight,
        TransformerMlpDownWeight,
        JointHiddenOutput,
      ]
    VideoFinalRmsNorm(_, _, _) =>
      [VideoFinalNormInput, FinalNormWeight, VideoFinalNormOutput]
    AudioFinalRmsNorm(_, _, _) =>
      [AudioFinalNormInput, FinalNormWeight, AudioFinalNormOutput]
    VideoFinalAdaLnF32Boundary(_) =>
      [
        VideoFinalNormOutput,
        FinalAdaLnShiftInput,
        FinalAdaLnScaleInput,
        VideoF32HeadInput,
      ]
    AudioFinalAdaLnF32Boundary(_) =>
      [
        AudioFinalNormOutput,
        FinalAdaLnShiftInput,
        FinalAdaLnScaleInput,
        AudioF32HeadInput,
      ]
    FinalAdaLnParameterRow(_, _) =>
      [
        FinalAdaLnConditioningInput,
        FinalAdaLnWeight,
        FinalAdaLnBias,
        FinalAdaLnShiftOutput,
        FinalAdaLnScaleOutput,
      ]
    PackedModalityAdaLnRows(_) =>
      [
        PackedAdaLnShiftInput,
        PackedAdaLnScaleInput,
        VideoPackedRowIndices,
        AudioPackedRowIndices,
        VideoAdaLnShiftOutput,
        VideoAdaLnScaleOutput,
        AudioAdaLnShiftOutput,
        AudioAdaLnScaleOutput,
      ]
    TimestepFrequencyRows(_) =>
      [DistinctTimestepValuesInput, TimestepFrequencyRowsOutput]
    TimestepEmbeddingMlpRows(_, _, _) =>
      [
        TimestepFrequencyRowsOutput,
        TimestepMlpWeight1,
        TimestepMlpBias1,
        TimestepMlpWeight2,
        TimestepMlpBias2,
        TimestepEmbeddingRowsOutput,
      ]
    DistinctTimestepAdaLnParameterRows(_, _) =>
      [
        TimestepEmbeddingRowsOutput,
        FinalAdaLnWeight,
        FinalAdaLnBias,
        FinalAdaLnShiftOutput,
        FinalAdaLnScaleOutput,
      ]
    VideoOutputProjection(_, _, _, _, _) =>
      [
        VideoF32HeadInput,
        ProjectionWeight,
        ProjectionBias,
        VideoPredictionOutput,
      ]
    AudioOutputProjection(_, _) =>
      [
        AudioF32HeadInput,
        ProjectionWeight,
        ProjectionBias,
        AudioPredictionOutput,
      ]
    ShiftedFlowStep(VideoLatent, _, _, _) =>
      [
        VideoLatentInput,
        VideoPredictionInput,
        FlowCoefficientsInput,
        VideoLatentOutput,
      ]
    ShiftedFlowStep(AudioLatent, _, _, _) =>
      [
        AudioLatentInput,
        AudioPredictionInput,
        FlowCoefficientsInput,
        AudioLatentOutput,
      ]
    VideoVaeLatentDestandardization(_, _, _) =>
      [
        VideoNormalizedLatentInput,
        VideoLatentStandardDeviationInput,
        VideoLatentMeanInput,
        VideoDestandardizedLatentOutput,
      ]
    VideoVaeDecoderBody(_, _, _, _) =>
      [VideoDestandardizedLatentOutput, ModelWeights, VideoMediaOutput]
    AudioVaeLatentDestandardization(_, _, _, _) =>
      [
        AudioNormalizedLatentInput,
        AudioLatentStandardDeviationInput,
        AudioLatentMeanInput,
        AudioDestandardizedLatentOutput,
      ]
    AudioVaeDecoderInputProjection(_, _, _, _, _, _) =>
      [
        AudioDestandardizedLatentOutput,
        AudioDecoderInputProjectionWeight,
        AudioDecoderInputProjectionBias,
        AudioDecoderInputProjectionOutput,
      ]
    AudioVaeDecoderConvPre(_, _, _, _, _, _, _, _, _) =>
      [
        AudioDecoderInputProjectionOutput,
        AudioDecoderConvPreWeightG,
        AudioDecoderConvPreWeightV,
        AudioDecoderConvPreBias,
        AudioDecoderConvPreNormalizedWeightWorkspace,
        AudioDecoderConvPreOutput,
      ]
    AudioVaeDecoderFirstUpsample(_, _, _, _, _, _, _, _, _) =>
      [
        AudioDecoderConvPreOutput,
        AudioDecoderFirstUpsampleWeightG,
        AudioDecoderFirstUpsampleWeightV,
        AudioDecoderFirstUpsampleBias,
        AudioDecoderFirstUpsampleNormalizedWeightWorkspace,
        AudioDecoderFirstUpsampleOutput,
      ]
    AudioVaeDecoderFirstAmpActivation(_, _, _, _, _, _) =>
      [
        AudioDecoderFirstUpsampleOutput,
        AudioDecoderFirstAmpAlpha,
        AudioDecoderFirstAmpBeta,
        AudioDecoderFirstAmpUpsampleFilter,
        AudioDecoderFirstAmpDownsampleFilter,
        AudioDecoderFirstAmpUpsampleWorkspace,
        AudioDecoderFirstAmpActivatedWorkspace,
        AudioDecoderFirstAmpActivationOutput,
      ]
    AudioVaeDecoderFirstAmpStage(_, _, _, _, _, _) =>
      [
        AudioDecoderFirstUpsampleOutput,
        AudioDecoderFirstAmpActivationOutput,
        ModelWeights,
        Workspace,
        AudioDecoderFirstAmpStageOutput,
      ]
    AudioVaeDecoderUpsampleTransition(stage, _, _, _, _, _, _, _, _, _) =>
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
    AudioVaeDecoderAmpActivation(stage, _, _, _, _, _, _) => {
      let input = if stage == 1 {
        AudioDecoderUpsampleTransitionOutput
      } else {
        AudioDecoderUpsampleTransition2Output
      }
      let output = if stage == 1 {
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
    AudioVaeDecoderAmpStage(_, _, _, _, _, _, _, _) =>
      [
        AudioDecoderUpsampleTransitionOutput,
        AudioDecoderAmpActivationOutput,
        ModelWeights,
        Workspace,
        AudioDecoderAmpStageOutput,
      ]
    AudioVaeDecoderBody(_, _, _, _) =>
      [AudioDecoderAmpActivation2Output, ModelWeights, AudioMediaOutput]
  }
}

///|
fn operands_are_exact(
  operation : @joint_diffusion_kernel_requirements.JointDiffusionSemanticOperation,
  operands : ReadOnlyArray[JointDiffusionLaunchOperand],
  maximum : Int,
) -> Bool {
  if operands.length() > maximum {
    return false
  }
  let expected = expected_operand_roles(operation)
  if operands.length() != expected.length() {
    return false
  }
  for index, operand in operands {
    if operand.role() != expected[index] {
      return false
    }
  }
  true
}
