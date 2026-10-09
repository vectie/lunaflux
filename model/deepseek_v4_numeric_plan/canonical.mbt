///|
fn hash_text(
  hasher : @canonical_sha256.CanonicalSha256,
  value : StringView,
) -> Unit {
  let bytes = @utf8.encode(value)
  ignore(hasher.update_u64_le(bytes.length().to_uint64()))
  ignore(hasher.update_bytes(bytes))
}

///|
fn hash_int(hasher : @canonical_sha256.CanonicalSha256, value : Int) -> Unit {
  ignore(hasher.update_u64_le(Int64::from_int(value).reinterpret_as_uint64()))
}

///|
fn profile_tag(profile : @deepseek_v4_spec.DeepSeekV4Profile) -> Byte {
  match profile {
    Flash => 1
    FlashBase => 2
    Pro => 3
    ProBase => 4
    Flash0731 => 5
  }
}

///|
fn storage_tag(storage : DeepSeekV4LogicalStorage) -> Byte {
  match storage {
    PlainBFloat16 => 1
    PlainFloat32 => 2
    PlainInt64 => 3
    Fp8E4M3Block128Parameter => 4
    Ue8M0Block128ScaleMetadata => 5
    PackedFp4E2M1Block32Parameter => 6
    Ue8M0Block32ScaleMetadata => 7
  }
}

///|
fn operation_kind_tag(kind : DeepSeekV4NumericOperationKind) -> Byte {
  match kind {
    TokenEmbedding => 1
    BlockRmsNorm => 2
    HyperConnectionBlockControl(AttentionSite) => 30
    HyperConnectionBlockControl(FeedForwardSite) => 31
    HyperConnectionPreReduce(AttentionSite) => 32
    HyperConnectionPreReduce(FeedForwardSite) => 33
    HyperConnectionPostCombine(AttentionSite) => 34
    HyperConnectionPostCombine(FeedForwardSite) => 35
    HyperConnectionHeadReduce => 36
    LowRankAttentionProjection => 5
    QueryKeyRmsNorm => 6
    ScaledRotaryPosition => 7
    SlidingWindowAttention => 8
    LearnedIndexSelection => 9
    IndexedCompressedAttention => 10
    DeterministicCompressedSelection => 11
    DeterministicWindowCompressedIndexJoinV1 => 37
    NonOverlappingCompressedKeyValuePrepareV1 => 38
    SelectedSparseAttentionV1 => 39
    AttentionOutputInverseRotaryV1 => 40
    AttentionOutputAProjectionV1 => 41
    AttentionOutputBProjectionV1 => 42
    MoeRouterProjection => 14
    ExpertScoreTransform => 15
    TokenHashRouting => 16
    StableTopKRouting => 17
    RoutedExpertClampedSwiGlu => 18
    SharedExpertClampedSwiGlu => 19
    RoutedSharedExpertCombine => 20
    MultiTokenPredictionInputProjection => 21
    DsparkTargetLayerRead => 22
    DsparkMainProjection => 23
    DsparkNoiseEmbedding => 24
    AuxiliaryOutputHead => 25
    DsparkMarkovAdjustment => 26
    DsparkConfidence => 27
    FinalRmsNorm => 28
    LanguageModelHead => 29
  }
}

///|
fn hash_scope(
  hasher : @canonical_sha256.CanonicalSha256,
  scope : DeepSeekV4NumericOperationScope,
) -> Unit {
  match scope {
    Global => ignore(hasher.update_byte(1))
    DecoderLayer(layer) => {
      ignore(hasher.update_byte(2))
      hash_int(hasher, layer)
    }
    PredictionStage(stage) => {
      ignore(hasher.update_byte(3))
      hash_int(hasher, stage)
    }
  }
}

///|
fn logical_manifest_digest(
  profile : @deepseek_v4_spec.DeepSeekV4Profile,
  identity : @spec.ModelIdentity,
  expected_tensor_count : Int,
  runs : ArrayView[DeepSeekV4NumericTensorRun],
  codebook : DeepSeekV4Fp4CodebookBinding?,
) -> @spec.PlanDigest {
  let hasher = @canonical_sha256.CanonicalSha256::new()
  hash_text(hasher, "lunaflux/deepseek-v4-logical-manifest/v1")
  ignore(hasher.update_byte(profile_tag(profile)))
  hash_text(hasher, identity.content().as_hex())
  hash_text(hasher, identity.plan().as_hex())
  hash_int(hasher, expected_tensor_count)
  hash_int(hasher, runs.length())
  for run in runs {
    hash_int(hasher, run.role.stable_index())
    ignore(hasher.update_byte(storage_tag(run.storage)))
    hash_int(hasher, run.tensor_count)
  }
  ignore(hasher.update_byte(if codebook is Some(_) { 1 } else { 0 }))
  try! @spec.PlanDigest::from_sha256(hasher.finalize_hex())
}

///|
fn hash_execution(
  hasher : @canonical_sha256.CanonicalSha256,
  execution : DeepSeekV4OperationExecution,
) -> Unit {
  match execution {
    NumericContract(contract) => {
      ignore(hasher.update_byte(1))
      hash_text(hasher, contract.digest().as_hex())
    }
    RequiresCheckedInt64ToInt32TokenHash => ignore(hasher.update_byte(2))
    UnsupportedPackedFp4E2M1 => ignore(hasher.update_byte(3))
  }
}

///|
fn numeric_schema_digest(
  identity : @spec.ModelIdentity,
  manifest : @spec.PlanDigest,
  operations : ArrayView[DeepSeekV4NumericOperationBinding],
) -> @spec.PlanDigest {
  let hasher = @canonical_sha256.CanonicalSha256::new()
  hash_text(hasher, "lunaflux/deepseek-v4-numeric-schema/v3")
  hash_text(hasher, identity.content().as_hex())
  hash_text(hasher, identity.plan().as_hex())
  hash_text(hasher, manifest.as_hex())
  hash_int(hasher, operations.length())
  for operation in operations {
    hash_int(hasher, operation.ordinal)
    hash_scope(hasher, operation.scope)
    ignore(hasher.update_byte(operation_kind_tag(operation.kind)))
    hash_execution(hasher, operation.execution)
  }
  try! @spec.PlanDigest::from_sha256(hasher.finalize_hex())
}

///|
fn numeric_binding_digest(
  identity : @spec.ModelIdentity,
  manifest : @spec.PlanDigest,
  schema : @spec.PlanDigest,
) -> @spec.PlanDigest {
  let hasher = @canonical_sha256.CanonicalSha256::new()
  hash_text(hasher, "lunaflux/deepseek-v4-numeric-binding/v3")
  hash_text(hasher, identity.content().as_hex())
  hash_text(hasher, identity.plan().as_hex())
  hash_text(hasher, manifest.as_hex())
  hash_text(hasher, schema.as_hex())
  try! @spec.PlanDigest::from_sha256(hasher.finalize_hex())
}
