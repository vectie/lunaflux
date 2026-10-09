///|
fn encode_audio_amp_activation(
  writer : RequirementCanonicalWriter,
  stage : Int,
  batch : Int,
  channels : Int,
  ratio : Int,
  kernel : Int,
  storage : JointDiffusionAudioVaeAliasFreeActivationStorage,
  layout : JointDiffusionAudioVaeLatentLayout,
) -> Unit raise JointDiffusionKernelRequirementsError {
  writer.byte(30)
  writer.int(stage)
  writer.int(batch)
  writer.int(channels)
  writer.int(ratio)
  writer.int(kernel)
  writer.byte(
    match storage {
      Float32InputAlphaBetaUpsampleFilterDownsampleFilterWorkspacesOutput => 1
    },
  )
  writer.byte(
    match layout {
      BatchChannelTime => 1
    },
  )
}

///|
fn encode_audio_amp_stage(
  writer : RequirementCanonicalWriter,
  stage : Int,
  batch : Int,
  channels : Int,
  blocks : Int,
  pairs : Int,
  schedule : JointDiffusionAudioVaeAmpSchedule,
  storage : JointDiffusionAudioVaeAmpStageStorage,
  layout : JointDiffusionAudioVaeLatentLayout,
) -> Unit raise JointDiffusionKernelRequirementsError {
  writer.byte(31)
  writer.int(stage)
  writer.int(batch)
  writer.int(channels)
  writer.int(blocks)
  writer.int(pairs)
  writer.byte(
    match schedule {
      Kernels3_7_11Dilations1_3_5 => 1
    },
  )
  writer.byte(
    match storage {
      Float32OriginalPrecomputedActivationPackedParametersWorkspaceOutput => 1
    },
  )
  writer.byte(
    match layout {
      BatchChannelTime => 1
    },
  )
}
