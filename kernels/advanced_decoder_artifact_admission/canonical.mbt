///|
priv struct AdmissionCanonicalWriter {
  hasher : @canonical_sha256.CanonicalSha256
  mut byte_count : Int
  maximum : Int
}

///|
fn AdmissionCanonicalWriter::new(maximum : Int) -> Self {
  { hasher: @canonical_sha256.CanonicalSha256::new(), byte_count: 0, maximum, }
}

///|
fn AdmissionCanonicalWriter::reserve(
  self : Self,
  count : Int,
) -> Unit raise AdvancedDecoderArtifactAdmissionError {
  if count < 0 || self.byte_count > self.maximum - count {
    raise AdmissionEncodingTooLarge
  }
  self.byte_count += count
}

///|
fn AdmissionCanonicalWriter::byte(
  self : Self,
  value : Byte,
) -> Unit raise AdvancedDecoderArtifactAdmissionError {
  self.reserve(1)
  ignore(self.hasher.update_byte(value))
}

///|
fn AdmissionCanonicalWriter::int(
  self : Self,
  value : Int,
) -> Unit raise AdvancedDecoderArtifactAdmissionError {
  self.reserve(8)
  ignore(
    self.hasher.update_u64_le(Int64::from_int(value).reinterpret_as_uint64()),
  )
}

///|
fn AdmissionCanonicalWriter::int64(
  self : Self,
  value : Int64,
) -> Unit raise AdvancedDecoderArtifactAdmissionError {
  self.reserve(8)
  ignore(self.hasher.update_u64_le(value.reinterpret_as_uint64()))
}

///|
fn AdmissionCanonicalWriter::text(
  self : Self,
  value : String,
) -> Unit raise AdvancedDecoderArtifactAdmissionError {
  let bytes = @utf8.encode(value)
  self.int(bytes.length())
  self.reserve(bytes.length())
  ignore(self.hasher.update_bytes(bytes))
}

///|
fn capability_tag(
  capability : @advanced_decoder_kernel_requirements.AdvancedDecoderAotCapability,
) -> Byte {
  match capability {
    EmbeddingLookupAot => 1
    LowRankAttentionProjectionAot => 2
    QueryKeyRmsNormAot => 3
    ScaledRotaryPositionAot => 4
    HyperConnectionSinkhornAot => 5
    HyperConnectionMixAot => 6
    SlidingWindowCausalAttentionAot => 7
    LearnedCompressedIndexAot => 8
    IndexedCompressedAttentionAot => 9
    DeterministicCompressedIndexAot => 10
    ExpertScoreTransformAot => 12
    TokenHashRoutingAot => 13
    StableTopKRoutingAot => 14
    RoutedExpertFeedForwardAot => 15
    SharedExpertFeedForwardAot => 16
    RoutedSharedExpertCombineAot => 17
    AuxiliaryProjectionAot => 18
    AuxiliaryDecoderStageAot => 19
    DsparkAot => 20
    RmsNormAot => 21
    DenseProjectionAot => 22
    SelectedExpertWeightFinalizeAot => 23
    HyperConnectionBlockControlV1Aot => 24
    HyperConnectionPreReduceV1Aot => 25
    HyperConnectionPostCombineV1Aot => 26
    HyperConnectionHeadReduceV1Aot => 27
    QueryLowRankAProjectionAot => 28
    QueryLowRankBProjectionAot => 29
    DeterministicWindowCompressedIndexJoinV1Aot => 30
    NonOverlappingCompressedKeyValuePrepareV1Aot => 31
    SelectedSparseAttentionV1Aot => 32
    AttentionOutputInverseRotaryV1Aot => 33
    AttentionOutputAProjectionV1Aot => 34
    AttentionOutputBProjectionV1Aot => 35
  }
}

///|
fn operand_role_tag(role : AdvancedDecoderLaunchOperandRole) -> Byte {
  match role {
    StepCountsInput => 18
    TokenIdsInput => 1
    PositionInput => 2
    HiddenInput => 3
    HiddenOutput => 4
    ModelWeights => 5
    HyperConnectionStateInput => 6
    HyperConnectionStateOutput => 7
    HyperConnectionFunctionWeightsInput => 28
    HyperConnectionBaseInput => 29
    HyperConnectionScaleInput => 30
    HyperConnectionPreOutput => 31
    HyperConnectionPostOutput => 32
    HyperConnectionCombinationOutput => 33
    HyperConnectionPreInput => 34
    HyperConnectionReducedOutput => 35
    HyperConnectionBranchInput => 36
    HyperConnectionPostInput => 37
    HyperConnectionCombinationInput => 38
    HyperConnectionCombinedOutput => 39
    RoutingScoresInput => 8
    RoutingScoresOutput => 9
    RoutingIndicesInput => 10
    RoutingIndicesOutput => 11
    CompressedQueryPositionsInput => 40
    CompressedCacheOffsetsInput => 41
    CompressedIndexCountsOutput => 42
    CompressedIndicesOutput => 43
    SparseAttentionQueryInput => 44
    SparseAttentionQueryBatchIdsInput => 45
    SparseAttentionKeyValueInput => 46
    SparseAttentionCacheLengthsInput => 47
    SparseAttentionSinkInput => 48
    SparseAttentionSelectedIndicesInput => 49
    SparseAttentionOutput => 50
    WindowQueryPositionsInput => 51
    WindowSequenceLengthsInput => 52
    WindowQueryModesInput => 53
    CompressedIndexCountsInput => 54
    CompressedIndicesInput => 55
    JoinedIndexCountsOutput => 56
    JoinedIndicesOutput => 57
    PreparedKeyValueModesInput => 58
    PreparedKeyValueSequenceLengthsInput => 59
    UncompressedKeyValueInput => 60
    CompressedKeyValueInput => 61
    PreparedKeyValueLengthsOutput => 62
    PreparedKeyValueOutput => 63
    AttentionOutputRotaryInOut => 64
    AttentionOutputProjectionInput => 65
    AttentionOutputAWeightsInput => 66
    AttentionOutputRankOutput => 67
    AttentionOutputRankInput => 68
    AttentionOutputBWeightCodesInput => 69
    AttentionOutputBWeightScalesInput => 70
    QueryAWeightCodesInput => 71
    QueryAWeightScalesInput => 72
    QueryLowRankAOutput => 73
    KeyValueWeightCodesInput => 74
    KeyValueWeightScalesInput => 75
    KeyValueProjectionOutput => 76
    QueryInput => 19
    QueryNormWeights => 20
    QueryOutput => 21
    KeyValueInput => 22
    KeyValueNormWeights => 23
    KeyValueOutput => 24
    TokenHashTableInput => 25
    RoutingSelectionBiasInput => 26
    RoutingWeightsOutput => 27
    ExpertInput => 12
    ExpertOutput => 13
    AuxiliaryInput => 14
    AuxiliaryOutput => 15
    LogitsOutput => 16
    Workspace => 17
  }
}

///|
fn encode_entry_point(
  writer : AdmissionCanonicalWriter,
  entry : @catalog.AotKernelEntryPoint,
) -> Unit raise AdvancedDecoderArtifactAdmissionError {
  writer.text(entry.family().module_digest().sha256())
  writer.int(entry.family().family_id().as_int())
  writer.int(entry.entry_point_id().as_int())
}

///|
fn encode_dimensions(
  writer : AdmissionCanonicalWriter,
  dimensions : @launch_contract.AotLaunchDimensions,
) -> Unit raise AdvancedDecoderArtifactAdmissionError {
  writer.int(dimensions.grid_x())
  writer.int(dimensions.grid_y())
  writer.int(dimensions.grid_z())
  writer.int(dimensions.block_x())
  writer.int(dimensions.block_y())
  writer.int(dimensions.block_z())
  writer.int(dimensions.shared_memory_bytes())
}

///|
fn canonical_admission_digest(
  source_identity : @spec.ModelIdentity,
  requirements_digest : @advanced_decoder_kernel_requirements.AdvancedDecoderKernelRequirementsDigest,
  target : @catalog.DeviceTarget,
  modules : ArrayView[AdvancedDecoderAdmittedModule],
  entries : ArrayView[AdvancedDecoderAdmittedEntryPoint],
  launches : ArrayView[AdvancedDecoderAdmittedLaunch],
  maximum : Int,
) -> AdvancedDecoderArtifactAdmissionDigest raise AdvancedDecoderArtifactAdmissionError {
  let writer = AdmissionCanonicalWriter::new(maximum)
  writer.text("lunaflux/advanced-decoder-artifact-admission/v1")
  writer.text(source_identity.content().as_hex())
  writer.text(source_identity.plan().as_hex())
  writer.text(requirements_digest.as_hex())
  writer.int(target.compute_major())
  writer.int(target.compute_minor())
  writer.byte(if target.supports_bf16() { 1 } else { 0 })
  writer.byte(if target.supports_cublas_lt() { 1 } else { 0 })
  writer.byte(1) // InertUnqualified is the only status.
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
    writer.text(launch.module_digest.sha256())
    encode_dimensions(writer, launch.dimensions)
    writer.int(launch.operands.length())
    for operand in launch.operands {
      writer.byte(operand_role_tag(operand.role()))
      writer.int64(operand.byte_count())
      writer.int64(operand.alignment())
    }
    match launch.declared_qualification {
      Some(declared) => {
        writer.byte(1)
        writer.text(declared.as_hex())
      }
      None => writer.byte(0)
    }
  }
  { hex: writer.hasher.finalize_hex(), }
}
