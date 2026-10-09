///|
fn hash_text(
  hasher : @canonical_sha256.CanonicalSha256,
  value : StringView,
) -> Unit {
  let encoded = @utf8.encode(value)
  ignore(hasher.update_u64_le(encoded.length().to_uint64()))
  ignore(hasher.update_bytes(encoded))
}

///|
fn tensor_role_tag(role : Glm53LogicalTensorRole) -> Byte {
  match role {
    BFloat16Parameter => 0x01
    Float32RoutingCorrection => 0x02
    Float32KdaStateParameter => 0x03
    Float32HyperConnectionParameter => 0x04
    Fp8Parameter => 0x05
    Fp8BlockScaleMetadata => 0x06
  }
}

///|
fn operation_kind_tag(kind : Glm53NumericOperationKind) -> Byte {
  match kind {
    TokenEmbedding => 0x01
    LayerNorm => 0x02
    HyperConnectionBf16Function => 0x03
    HyperConnectionF32Control => 0x04
    DsaLowRankProjection => 0x05
    DsaIndexerProjection => 0x06
    DsaIndexPooling => 0x07
    DsaIndexSelection => 0x08
    DsaIndexReuse => 0x09
    DsaSparseAttention => 0x0a
    AttentionOutputProjection => 0x0b
    KdaProjection => 0x0c
    KdaShortConvolution => 0x0d
    KdaF32DecayControl => 0x0e
    KdaRecurrentUpdate => 0x0f
    DenseSwiGlu => 0x10
    MoeRouterProjection => 0x11
    MoeF32SigmoidAndCorrection => 0x12
    RoutedExpertSwiGlu => 0x13
    SharedExpertSwiGlu => 0x14
    MtpFusionProjection => 0x15
    SharedHeadNorm => 0x16
    FinalNorm => 0x17
    LanguageModelHead => 0x18
    VisionPatchProjection => 0x19
    VisionQkvProjection => 0x1a
    VisionAttention => 0x1b
    VisionOutputProjection => 0x1c
    VisionSwiGlu => 0x1d
    VisionSpatialMerge => 0x1e
    VisionProjector => 0x1f
    KdaOutputGateProjection => 0x20
    KdaGatedRmsNorm => 0x21
    KdaOutputProjection => 0x22
    MoeF32GroupedRouting => 0x23
    RoutedSharedExpertBf16Combine => 0x24
  }
}

///|
fn hash_scope(
  hasher : @canonical_sha256.CanonicalSha256,
  scope : Glm53NumericOperationScope,
) -> Unit {
  match scope {
    Global => ignore(hasher.update_byte(0x01))
    DecoderLayer(layer_index~) => {
      ignore(hasher.update_byte(0x02))
      ignore(hasher.update_u64_le(layer_index.to_uint64()))
    }
    AuxiliaryLayer(stage_index~) => {
      ignore(hasher.update_byte(0x03))
      ignore(hasher.update_u64_le(stage_index.to_uint64()))
    }
    VisionFrontend => ignore(hasher.update_byte(0x04))
    VisionLayer(layer_index~) => {
      ignore(hasher.update_byte(0x05))
      ignore(hasher.update_u64_le(layer_index.to_uint64()))
    }
    VisionProjector => ignore(hasher.update_byte(0x06))
  }
}

///|
fn numeric_binding_digest(
  profile : @glm53_spec.Glm53ProfileId,
  encoding : @glm53_weights.Glm53WeightEncoding,
  execution_identity : @spec.ModelIdentity,
  manifest_layout_digest : @spec.PlanDigest,
  schema : @numeric_contract.ModelNumericSchema,
  tensors : ArrayView[Glm53NumericTensorBinding],
  operations : ArrayView[Glm53NumericOperationBinding],
) -> @spec.PlanDigest {
  let hasher = @canonical_sha256.CanonicalSha256::new()
  hash_text(hasher, "lunaflux.glm53.numeric-plan.v1")
  ignore(
    hasher.update_byte(
      match profile {
        FullText => 0x01
        FlashMultimodal => 0x02
      },
    ),
  )
  ignore(
    hasher.update_byte(
      match encoding {
        BFloat16WithFloat32Routing => 0x01
        Fp8E4M3Block128Dynamic => 0x02
      },
    ),
  )
  hash_text(hasher, execution_identity.content().as_hex())
  hash_text(hasher, execution_identity.plan().as_hex())
  hash_text(hasher, manifest_layout_digest.as_hex())
  hash_text(hasher, schema.digest().as_hex())
  ignore(hasher.update_u64_le(tensors.length().to_uint64()))
  for tensor in tensors {
    ignore(hasher.update_u64_le(tensor.ordinal.to_uint64()))
    ignore(hasher.update_byte(tensor_role_tag(tensor.role)))
  }
  ignore(hasher.update_u64_le(operations.length().to_uint64()))
  for operation in operations {
    ignore(hasher.update_u64_le(operation.ordinal.to_uint64()))
    hash_scope(hasher, operation.scope)
    ignore(hasher.update_byte(operation_kind_tag(operation.kind)))
    hash_text(hasher, operation.execution.digest().as_hex())
  }
  try! @spec.PlanDigest::from_sha256(hasher.finalize_hex())
}
