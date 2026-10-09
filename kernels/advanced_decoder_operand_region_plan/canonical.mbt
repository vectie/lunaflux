///|
priv struct OperandRegionCanonicalWriter {
  hasher : @canonical_sha256.CanonicalSha256
  mut byte_count : Int
  maximum : Int
}

///|
fn OperandRegionCanonicalWriter::new(maximum : Int) -> Self {
  { hasher: @canonical_sha256.CanonicalSha256::new(), byte_count: 0, maximum, }
}

///|
fn OperandRegionCanonicalWriter::reserve(
  self : Self,
  count : Int,
) -> Unit raise AdvancedDecoderOperandRegionPlanError {
  if count < 0 || self.byte_count > self.maximum - count {
    raise CanonicalEncodingTooLarge
  }
  self.byte_count += count
}

///|
fn OperandRegionCanonicalWriter::byte(
  self : Self,
  value : Byte,
) -> Unit raise AdvancedDecoderOperandRegionPlanError {
  self.reserve(1)
  ignore(self.hasher.update_byte(value))
}

///|
fn OperandRegionCanonicalWriter::int(
  self : Self,
  value : Int,
) -> Unit raise AdvancedDecoderOperandRegionPlanError {
  self.reserve(8)
  ignore(
    self.hasher.update_u64_le(Int64::from_int(value).reinterpret_as_uint64()),
  )
}

///|
fn OperandRegionCanonicalWriter::int64(
  self : Self,
  value : Int64,
) -> Unit raise AdvancedDecoderOperandRegionPlanError {
  self.reserve(8)
  ignore(self.hasher.update_u64_le(value.reinterpret_as_uint64()))
}

///|
fn OperandRegionCanonicalWriter::text(
  self : Self,
  value : String,
) -> Unit raise AdvancedDecoderOperandRegionPlanError {
  let encoded = @utf8.encode(value)
  self.int(encoded.length())
  self.reserve(encoded.length())
  ignore(self.hasher.update_bytes(encoded))
}

///|
fn operand_region_role_tag(
  role : @advanced_decoder_artifact_admission.AdvancedDecoderLaunchOperandRole,
) -> Byte {
  match role {
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
    ExpertInput => 12
    ExpertOutput => 13
    AuxiliaryInput => 14
    AuxiliaryOutput => 15
    LogitsOutput => 16
    Workspace => 17
    StepCountsInput => 18
    QueryInput => 19
    QueryNormWeights => 20
    QueryOutput => 21
    KeyValueInput => 22
    KeyValueNormWeights => 23
    KeyValueOutput => 24
    TokenHashTableInput => 25
    RoutingSelectionBiasInput => 26
    RoutingWeightsOutput => 27
  }
}

///|
fn canonical_operand_region_plan_digest(
  regions : ArrayView[AdvancedDecoderOperandRegion],
  maximum : Int,
) -> AdvancedDecoderOperandRegionPlanDigest raise AdvancedDecoderOperandRegionPlanError {
  let writer = OperandRegionCanonicalWriter::new(maximum)
  writer.text("lunaflux/advanced-decoder-operand-region-plan/v1")
  writer.byte(1) // InertUnqualified.
  writer.int(regions.length())
  for region in regions {
    writer.int(region.operand_ordinal)
    writer.byte(operand_region_role_tag(region.role))
    writer.int64(region.byte_count)
    writer.int64(region.alignment)
    writer.text(region.allocation_identity_sha256)
    writer.int64(region.device_offset)
    writer.int64(region.end_offset)
    writer.int64(region.allocation_capacity_bytes)
  }
  AdvancedDecoderOperandRegionPlanDigest(writer.hasher.finalize_hex())
}
