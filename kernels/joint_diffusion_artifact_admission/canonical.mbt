///|
priv struct AdmissionCanonicalWriter {
  hasher : @canonical_sha256.CanonicalSha256
  mut byte_count : Int
}

///|
fn AdmissionCanonicalWriter::new() -> Self {
  { hasher: @canonical_sha256.CanonicalSha256::new(), byte_count: 0, }
}

///|
fn AdmissionCanonicalWriter::reserve(
  self : Self,
  count : Int,
) -> Unit raise JointDiffusionArtifactAdmissionError {
  if count < 0 || self.byte_count > 262144 - count {
    raise AdmissionEncodingTooLarge
  }
  self.byte_count += count
}

///|
fn AdmissionCanonicalWriter::byte(
  self : Self,
  value : Byte,
) -> Unit raise JointDiffusionArtifactAdmissionError {
  self.reserve(1)
  if !self.hasher.update_byte(value) {
    abort("finalized joint diffusion artifact admission hash")
  }
}

///|
fn AdmissionCanonicalWriter::int(
  self : Self,
  value : Int,
) -> Unit raise JointDiffusionArtifactAdmissionError {
  self.reserve(8)
  if !self.hasher.update_u64_le(Int64::from_int(value).reinterpret_as_uint64()) {
    abort("finalized joint diffusion artifact admission hash")
  }
}

///|
fn AdmissionCanonicalWriter::int64(
  self : Self,
  value : Int64,
) -> Unit raise JointDiffusionArtifactAdmissionError {
  self.reserve(8)
  if !self.hasher.update_u64_le(value.reinterpret_as_uint64()) {
    abort("finalized joint diffusion artifact admission hash")
  }
}

///|
fn AdmissionCanonicalWriter::text(
  self : Self,
  value : String,
) -> Unit raise JointDiffusionArtifactAdmissionError {
  let bytes = @utf8.encode(value)
  self.int(bytes.length())
  self.reserve(bytes.length())
  if !self.hasher.update_bytes(bytes) {
    abort("finalized joint diffusion artifact admission hash")
  }
}

///|
fn capability_tag(
  capability : @joint_diffusion_kernel_requirements.JointDiffusionAotCapability,
) -> Byte {
  match capability {
    TokenTextConditioningAot => 1
    OrderedKeyframeConditioningAot => 2
    OrderedReferenceConditioningAot => 3
    VideoLatentProjectionAot => 4
    AudioLatentProjectionAot => 5
    JointTransformerAttentionAot => 6
    JointTransformerMlpAot => 7
    ShiftedFlowStepAot => 8
    VideoVaeLatentDestandardizationAot => 9
    VideoVaeDecoderBodyAot => 22
    AudioVaeLatentDestandardizationAot => 10
    AudioVaeDecoderInputProjectionAot => 23
    AudioVaeDecoderConvPreAot => 24
    AudioVaeDecoderFirstUpsampleAot => 25
    AudioVaeDecoderFirstAmpActivationAot => 27
    AudioVaeDecoderFirstAmpStageAot => 28
    AudioVaeDecoderUpsampleTransitionAot => 29
    AudioVaeDecoderAmpActivationAot => 30
    AudioVaeDecoderAmpStageAot => 31
    AudioVaeDecoderBodyAot => 26
    VideoOutputProjectionAot => 11
    AudioOutputProjectionAot => 12
    VideoFinalRmsNormAot => 13
    AudioFinalRmsNormAot => 14
    VideoFinalAdaLnF32BoundaryAot => 15
    AudioFinalAdaLnF32BoundaryAot => 16
    FinalAdaLnParameterRowAot => 17
    PackedModalityAdaLnRowsAot => 18
    TimestepFrequencyRowsAot => 19
    TimestepEmbeddingMlpRowsAot => 20
    DistinctTimestepAdaLnParameterRowsAot => 21
  }
}

///|
fn operand_role_tag(role : JointDiffusionLaunchOperandRole) -> Byte {
  match role {
    TokenIdsInput => 1
    TextConditioningOutput => 2
    KeyframeMediaInput => 3
    OrderedReferenceMediaInput => 4
    OrderedConditioningOutput => 5
    VideoLatentInput => 6
    AudioLatentInput => 7
    VideoPredictionInput => 8
    AudioPredictionInput => 9
    FlowCoefficientsInput => 118
    JointHiddenInput => 11
    JointHiddenOutput => 12
    ModelWeights => 13
    VideoLatentOutput => 14
    AudioLatentOutput => 15
    VideoMediaOutput => 16
    AudioMediaOutput => 17
    Workspace => 18
    ProjectionWeight => 19
    ProjectionBias => 20
    VideoHiddenInput => 21
    AudioHiddenInput => 22
    VideoPredictionOutput => 23
    AudioPredictionOutput => 24
    VideoFinalNormInput => 25
    AudioFinalNormInput => 26
    FinalNormWeight => 27
    VideoFinalNormOutput => 28
    AudioFinalNormOutput => 29
    FinalAdaLnShiftInput => 30
    FinalAdaLnScaleInput => 31
    VideoF32HeadInput => 32
    AudioF32HeadInput => 33
    FinalAdaLnConditioningInput => 34
    FinalAdaLnWeight => 35
    FinalAdaLnBias => 36
    FinalAdaLnShiftOutput => 37
    FinalAdaLnScaleOutput => 38
    PackedAdaLnShiftInput => 39
    PackedAdaLnScaleInput => 40
    VideoPackedRowIndices => 41
    AudioPackedRowIndices => 42
    VideoAdaLnShiftOutput => 43
    VideoAdaLnScaleOutput => 44
    AudioAdaLnShiftOutput => 45
    AudioAdaLnScaleOutput => 46
    DistinctTimestepValuesInput => 47
    TimestepFrequencyRowsOutput => 48
    TimestepMlpWeight1 => 49
    TimestepMlpBias1 => 50
    TimestepMlpWeight2 => 51
    TimestepMlpBias2 => 52
    TimestepEmbeddingRowsOutput => 53
    TransformerMlpPackedGateUpWeight => 54
    TransformerMlpDownWeight => 55
    AttentionQueryWeight => 56
    AttentionKeyWeight => 57
    AttentionValueWeight => 58
    AttentionOutputWeight => 59
    AttentionQueryNormWeight => 60
    AttentionKeyNormWeight => 61
    AttentionRotaryCosInput => 62
    AttentionRotarySinInput => 63
    AttentionQueryBuffer => 64
    AttentionKeyBuffer => 65
    AttentionValueBuffer => 66
    AttentionNormalizedQueryBuffer => 67
    AttentionNormalizedKeyBuffer => 68
    AttentionOutputBuffer => 69
    VideoNormalizedLatentInput => 70
    VideoLatentStandardDeviationInput => 71
    VideoLatentMeanInput => 72
    VideoDestandardizedLatentOutput => 73
    AudioNormalizedLatentInput => 74
    AudioLatentStandardDeviationInput => 75
    AudioLatentMeanInput => 76
    AudioDestandardizedLatentOutput => 77
    AudioDecoderInputProjectionWeight => 78
    AudioDecoderInputProjectionBias => 79
    AudioDecoderInputProjectionOutput => 80
    AudioDecoderConvPreWeightG => 81
    AudioDecoderConvPreWeightV => 82
    AudioDecoderConvPreBias => 83
    AudioDecoderConvPreNormalizedWeightWorkspace => 84
    AudioDecoderConvPreOutput => 85
    AudioDecoderFirstUpsampleWeightG => 86
    AudioDecoderFirstUpsampleWeightV => 87
    AudioDecoderFirstUpsampleBias => 88
    AudioDecoderFirstUpsampleNormalizedWeightWorkspace => 89
    AudioDecoderFirstUpsampleOutput => 90
    AudioDecoderFirstAmpAlpha => 91
    AudioDecoderFirstAmpBeta => 92
    AudioDecoderFirstAmpUpsampleFilter => 93
    AudioDecoderFirstAmpDownsampleFilter => 94
    AudioDecoderFirstAmpUpsampleWorkspace => 95
    AudioDecoderFirstAmpActivatedWorkspace => 96
    AudioDecoderFirstAmpActivationOutput => 97
    AudioDecoderFirstAmpStageOutput => 98
    AudioDecoderUpsampleTransitionWeightG => 99
    AudioDecoderUpsampleTransitionWeightV => 100
    AudioDecoderUpsampleTransitionBias => 101
    AudioDecoderUpsampleTransitionNormalizedWeightWorkspace => 102
    AudioDecoderUpsampleTransitionOutput => 103
    AudioDecoderAmpAlpha => 104
    AudioDecoderAmpBeta => 105
    AudioDecoderAmpUpsampleFilter => 106
    AudioDecoderAmpDownsampleFilter => 107
    AudioDecoderAmpUpsampleWorkspace => 108
    AudioDecoderAmpActivatedWorkspace => 109
    AudioDecoderAmpActivationOutput => 110
    AudioDecoderAmpStageOutput => 111
    AudioDecoderUpsampleTransition2WeightG => 112
    AudioDecoderUpsampleTransition2WeightV => 113
    AudioDecoderUpsampleTransition2Bias => 114
    AudioDecoderUpsampleTransition2NormalizedWeightWorkspace => 115
    AudioDecoderUpsampleTransition2Output => 116
    AudioDecoderAmpActivation2Output => 117
  }
}

///|
fn encode_entry_point(
  writer : AdmissionCanonicalWriter,
  entry : @catalog.AotKernelEntryPoint,
) -> Unit raise JointDiffusionArtifactAdmissionError {
  writer.text(entry.family().module_digest().sha256())
  writer.int(entry.family().family_id().as_int())
  writer.int(entry.entry_point_id().as_int())
}

///|
fn canonical_admission_digest(
  source_identity : @spec.ModelIdentity,
  requirements_digest : @joint_diffusion_kernel_requirements.JointDiffusionKernelRequirementsDigest,
  target : @catalog.DeviceTarget,
  modules : ArrayView[JointDiffusionAdmittedModule],
  entries : ArrayView[JointDiffusionAdmittedEntryPoint],
  launches : ArrayView[JointDiffusionAdmittedLaunch],
) -> JointDiffusionArtifactAdmissionDigest raise JointDiffusionArtifactAdmissionError {
  let writer = AdmissionCanonicalWriter::new()
  writer.text("lunaflux/joint-diffusion-artifact-admission/v1")
  writer.text(source_identity.content().as_hex())
  writer.text(source_identity.plan().as_hex())
  writer.text(requirements_digest.as_hex())
  writer.int(target.compute_major())
  writer.int(target.compute_minor())
  writer.byte(if target.supports_bf16() { 1 } else { 0 })
  writer.byte(if target.supports_cublas_lt() { 1 } else { 0 })
  writer.int(modules.length())
  for artifact_module in modules {
    writer.text(artifact_module.digest.sha256())
    writer.int64(artifact_module.module_bytes.length().to_int64())
  }
  writer.int(entries.length())
  for entry in entries {
    encode_entry_point(writer, entry.identity)
    writer.int(entry.module_index)
    writer.text(entry.function_symbol)
  }
  writer.int(launches.length())
  for launch in launches {
    writer.int(launch.requirement_ordinal)
    writer.byte(capability_tag(launch.capability))
    encode_entry_point(writer, launch.entry_point)
    let dimensions = launch.dimensions
    writer.int(dimensions.grid_x())
    writer.int(dimensions.grid_y())
    writer.int(dimensions.grid_z())
    writer.int(dimensions.block_x())
    writer.int(dimensions.block_y())
    writer.int(dimensions.block_z())
    writer.int(dimensions.shared_memory_bytes())
    writer.int(launch.operands.length())
    for operand in launch.operands {
      writer.byte(operand_role_tag(operand.role()))
      writer.int64(operand.byte_count())
      writer.int64(operand.alignment())
    }
  }
  { hex: writer.hasher.finalize_hex(), }
}
