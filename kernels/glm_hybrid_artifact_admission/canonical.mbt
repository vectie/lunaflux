///|
priv struct AdmissionWriter {
  hash : @canonical_sha256.CanonicalSha256
  mut bytes : Int
  maximum : Int
}

///|
fn AdmissionWriter::new(maximum : Int) -> Self {
  { hash: @canonical_sha256.CanonicalSha256::new(), bytes: 0, maximum, }
}

///|
fn AdmissionWriter::reserve(
  self : Self,
  count : Int,
) -> Unit raise GlmHybridArtifactAdmissionError {
  if count < 0 || self.bytes > self.maximum - count {
    raise AdmissionEncodingTooLarge
  }
  self.bytes += count
}

///|
fn AdmissionWriter::byte(
  self : Self,
  value : Byte,
) -> Unit raise GlmHybridArtifactAdmissionError {
  self.reserve(1)
  ignore(self.hash.update_byte(value))
}

///|
fn AdmissionWriter::int(
  self : Self,
  value : Int,
) -> Unit raise GlmHybridArtifactAdmissionError {
  self.reserve(8)
  ignore(
    self.hash.update_u64_le(Int64::from_int(value).reinterpret_as_uint64()),
  )
}

///|
fn AdmissionWriter::int64(
  self : Self,
  value : Int64,
) -> Unit raise GlmHybridArtifactAdmissionError {
  self.reserve(8)
  ignore(self.hash.update_u64_le(value.reinterpret_as_uint64()))
}

///|
fn AdmissionWriter::text(
  self : Self,
  value : String,
) -> Unit raise GlmHybridArtifactAdmissionError {
  let data = @utf8.encode(value)
  self.int(data.length())
  self.reserve(data.length())
  ignore(self.hash.update_bytes(data))
}

///|
fn encode_launch(
  w : AdmissionWriter,
  value : @launch_contract.AotLaunchDimensions,
) -> Unit raise GlmHybridArtifactAdmissionError {
  w.int(value.grid_x())
  w.int(value.grid_y())
  w.int(value.grid_z())
  w.int(value.block_x())
  w.int(value.block_y())
  w.int(value.block_z())
  w.int(value.shared_memory_bytes())
}

///|
fn encode_operand_role(
  w : AdmissionWriter,
  role : GlmHybridLaunchOperandRole,
) -> Unit raise GlmHybridArtifactAdmissionError {
  w.byte(
    match role {
      StepCountsInput => 1
      TokenIdsInput => 2
      HiddenInput => 3
      ModelWeights => 4
      HiddenOutput => 5
      LogitsOutput => 6
      DsaIndexKeysInput => 7
      DsaIndexGateScoresInput => 8
      DsaIndexValidityInput => 9
      DsaPooledKeysOutput => 10
      DsaPooledRawIndicesOutput => 11
      DsaPooledValidityOutput => 12
      DsaPooledCountOutput => 13
      QueryPositionsInput => 14
      DsaIndexQueriesInput => 15
      DsaIndexHeadWeightsInput => 16
      DsaCandidateKeysInput => 17
      DsaCandidateRawIndicesInput => 18
      DsaCandidateValidityInput => 19
      DsaCandidateCountInput => 20
      DsaKeyValidityInput => 21
      DsaSelectedIndicesInput => 22
      DsaSelectedIndicesOutput => 23
      DsaKeyCountInput => 24
      DsaAttentionQueriesInput => 25
      DsaAttentionKeysInput => 26
      DsaAttentionValuesInput => 27
      DsaAttentionOutput => 28
      DsaQueryCompressedOutput => 29
      DsaKeyValueCompressedRopeOutput => 30
      DsaQueryCompressedInput => 31
      DsaQueryNormalizedOutput => 32
      DsaQueryHeadsOutput => 33
      DsaQueryNormalizedInput => 34
      DsaKeyValueNormalizedInput => 35
      DsaKeyValueHeadsOutput => 36
      DsaKeyValueCompressedRopeInput => 37
      DsaKeyValueNormalizedOutput => 38
      DsaKeyRopeSuffixOutput => 39
      DsaAttentionKeyRopeInput => 40
      DenseGateWeights => 41
      DenseUpWeights => 42
      DenseDownWeights => 43
      KdaSequenceOffsetsInput => 44
      KdaProjectedInput => 45
      KdaShortConvolutionWeights => 46
      KdaConvolutionInitialStateInput => 47
      KdaConvolvedOutput => 48
      KdaConvolutionFinalStateOutput => 49
      KdaRecurrentQueryInput => 50
      KdaRecurrentKeyInput => 51
      KdaRecurrentValueInput => 52
      KdaLogDecayInput => 53
      KdaInputGateInput => 54
      KdaRecurrentInitialStateInput => 55
      KdaRecurrentOutput => 56
      KdaRecurrentFinalStateOutput => 57
      KdaQueryProjectionWeights => 58
      KdaKeyProjectionWeights => 59
      KdaValueProjectionWeights => 60
      KdaForgetLowRankWeights => 61
      KdaForgetExpansionWeights => 62
      KdaInputGateWeights => 63
      KdaProjectedQueryOutput => 64
      KdaProjectedKeyOutput => 65
      KdaProjectedValueOutput => 66
      KdaForgetLowRankOutput => 67
      KdaRawForgetOutput => 68
      KdaBetaOutput => 69
      KdaOutputGateLowRankWeights => 70
      KdaOutputGateExpansionWeights => 71
      KdaOutputGateLowRankOutput => 72
      KdaOutputGateOutput => 73
      KdaGatedNormRecurrentInput => 74
      KdaGatedNormGateInput => 75
      KdaGatedNormWeights => 76
      KdaGatedNormOutput => 77
      KdaOutputProjectionInput => 78
      KdaOutputProjectionWeights => 79
      KdaOutputProjectionOutput => 80
      KdaDecayRawForgetInput => 81
      KdaDecayBiasInput => 82
      KdaDecayLogRateInput => 83
      KdaDecayOutput => 84
      MoeRouterFloat32Input => 85
      MoeRouterBf16Weights => 86
      MoeRouterFloat32LogitsOutput => 87
      MoeCorrectionBiasInput => 88
      MoeUncorrectedScoresOutput => 89
      MoeChoiceScoresOutput => 90
      MoeSelectedIndicesOutput => 91
      MoeSelectedWeightsOutput => 92
      MoeFeedForwardInput => 93
      MoeExpertGateWeights => 94
      MoeExpertUpWeights => 95
      MoeExpertDownWeights => 96
      MoeRoutedOutput => 97
      MoeSharedGateWeights => 98
      MoeSharedUpWeights => 99
      MoeSharedDownWeights => 100
      MoeSharedOutput => 101
      MoeCombinedOutput => 102
      DecoderResidualInput => 103
      DecoderResidualOutput => 104
      HyperPostControlInput => 105
      HyperCombineControlInput => 106
      HyperResidualStreamsInput => 107
      HyperSublayerOutputInput => 108
      HyperMixedStreamsOutput => 109
      HyperFunctionWeightsInput => 110
      HyperBaseInput => 111
      HyperScaleInput => 112
      HyperInitialCombinationOutput => 113
      HyperCollapsedOutput => 114
    },
  )
}

///|
fn encode_operands(
  w : AdmissionWriter,
  operands : ReadOnlyArray[GlmHybridLaunchOperand],
) -> Unit raise GlmHybridArtifactAdmissionError {
  w.int(operands.length())
  for operand in operands {
    encode_operand_role(w, operand.role)
    w.int64(operand.byte_count)
    w.int64(operand.alignment)
  }
}

///|
fn admission_digest(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
  target : @catalog.DeviceTarget,
  status : GlmHybridArtifactAdmissionStatus,
  artifacts : ArrayView[GlmHybridAdmittedArtifact],
  maximum : Int,
) -> GlmHybridArtifactAdmissionDigest raise GlmHybridArtifactAdmissionError {
  let w = AdmissionWriter::new(maximum)
  w.text("lunaflux/glm-hybrid-artifact-admission/v2")
  w.text(requirements.digest().as_hex())
  w.text(requirements.source_identity().content().as_hex())
  w.text(requirements.source_identity().plan().as_hex())
  w.int(target.compute_major())
  w.int(target.compute_minor())
  w.byte(if target.supports_bf16() { 1 } else { 0 })
  w.byte(if target.supports_cublas_lt() { 1 } else { 0 })
  w.byte(
    match status {
      NonRunnableUnqualified => 1
      NonRunnableWithDeclaredQualificationDigests => 2
    },
  )
  w.int(artifacts.length())
  for artifact in artifacts {
    w.int(artifact.ordinal)
    w.text(artifact.module_digest.sha256())
    w.int(artifact.module_byte_length)
    w.int(artifact.entry_point.family().family_id().as_int())
    w.int(artifact.entry_point.entry_point_id().as_int())
    w.text(artifact.function_symbol)
    encode_launch(w, artifact.launch)
    encode_operands(w, artifact.operands)
    match artifact.declared_qualification {
      Some(q) => {
        w.byte(1)
        w.text(q.as_hex())
      }
      None => w.byte(0)
    }
  }
  { hex: w.hash.finalize_hex(), }
}
