///|
priv struct Writer {
  hash : @canonical_sha256.CanonicalSha256
  mut bytes : Int
}

///|
fn Writer::new() -> Self {
  { hash: @canonical_sha256.CanonicalSha256::new(), bytes: 0, }
}

///|
fn Writer::reserve(
  self : Self,
  count : Int,
) -> Unit raise GlmHybridKernelRequirementsError {
  if count < 0 || self.bytes > 65536 - count {
    raise RequirementsEncodingTooLarge
  }
  self.bytes += count
}

///|
fn Writer::byte(
  self : Self,
  value : Byte,
) -> Unit raise GlmHybridKernelRequirementsError {
  self.reserve(1)
  ignore(self.hash.update_byte(value))
}

///|
fn Writer::int(
  self : Self,
  value : Int,
) -> Unit raise GlmHybridKernelRequirementsError {
  self.reserve(8)
  ignore(
    self.hash.update_u64_le(Int64::from_int(value).reinterpret_as_uint64()),
  )
}

///|
fn Writer::double(
  self : Self,
  value : Double,
) -> Unit raise GlmHybridKernelRequirementsError {
  self.reserve(8)
  ignore(self.hash.update_u64_le(value.reinterpret_as_uint64()))
}

///|
fn Writer::bool(
  self : Self,
  value : Bool,
) -> Unit raise GlmHybridKernelRequirementsError {
  self.byte(if value { 1 } else { 0 })
}

///|
fn Writer::text(
  self : Self,
  value : String,
) -> Unit raise GlmHybridKernelRequirementsError {
  let data = @utf8.encode(value)
  self.int(data.length())
  self.reserve(data.length())
  ignore(self.hash.update_bytes(data))
}

///|
fn Writer::optional_double(
  self : Self,
  value : Double?,
) -> Unit raise GlmHybridKernelRequirementsError {
  match value {
    Some(v) => {
      self.byte(1)
      self.double(v)
    }
    None => self.byte(0)
  }
}

///|
fn encode_score(
  writer : Writer,
  value : @advanced_decoder_execution_plan.ExpertScoreFunction,
) -> Unit raise GlmHybridKernelRequirementsError {
  writer.byte(
    match value {
      SqrtSoftplus => 1
      Sigmoid => 2
    },
  )
}

///|
fn encode_activation(
  writer : Writer,
  value : @advanced_decoder_execution_plan.ExpertActivation,
) -> Unit raise GlmHybridKernelRequirementsError {
  match value {
    SwiGlu => writer.byte(1)
    ClampedSwiGlu(limit) => {
      writer.byte(2)
      writer.double(limit)
    }
  }
}

///|
fn encode_operation(
  w : Writer,
  op : GlmHybridSemanticOperation,
) -> Unit raise GlmHybridKernelRequirementsError {
  match op {
    TokenEmbedding => w.byte(1)
    DecoderRmsNorm(a, b) => {
      w.byte(2)
      w.int(a)
      w.double(b)
    }
    QueryLowRankProjection(a) => {
      w.byte(3)
      w.int(a)
    }
    KeyValueLowRankProjection(a, b, c, d) => {
      w.byte(4)
      w.int(a)
      w.int(b)
      w.int(c)
      w.int(d)
    }
    QueryKeyRmsNorm(a) => {
      w.byte(5)
      w.double(a)
    }
    GlmRotaryPosition(a, b) => {
      w.byte(6)
      w.optional_double(a)
      w.bool(b)
    }
    DsaIndexCompute(a, b, c, d) => {
      w.byte(7)
      w.int(a)
      w.int(b)
      w.int(c)
      w.bool(d)
    }
    DsaIndexPooling(a, b, c) => {
      w.byte(8)
      w.int(a)
      w.bool(b)
      w.bool(c)
    }
    DsaIndexReuse => w.byte(9)
    DsaSparseAttention(a, b, c, d) => {
      w.byte(10)
      w.int(a)
      w.int(b)
      w.int(c)
      w.int(d)
    }
    AttentionOutputProjection(a, b) => {
      w.byte(11)
      w.int(a)
      w.int(b)
    }
    KdaProjection(a, b) => {
      w.byte(12)
      w.int(a)
      w.int(b)
    }
    KdaShortConvolution(a, b) => {
      w.byte(13)
      w.int(a)
      w.int(b)
    }
    KdaRecurrentUpdate(a, b, c) => {
      w.byte(14)
      w.int(a)
      w.int(b)
      w.double(c)
    }
    KdaOutputProjection(a, b) => {
      w.byte(15)
      w.int(a)
      w.int(b)
    }
    HyperConnectionFunction(a, b, c, d) => {
      w.byte(16)
      w.int(a)
      w.int(b)
      w.double(c)
      w.double(d)
    }
    HyperConnectionSinkhorn(a, b, c) => {
      w.byte(17)
      w.int(a)
      w.int(b)
      w.double(c)
    }
    HyperConnectionMix(a) => {
      w.byte(18)
      w.int(a)
    }
    DenseSwiGlu(a, b) => {
      w.byte(19)
      w.int(a)
      w.int(b)
    }
    MoeRouterProjection(a, b, c) => {
      w.byte(20)
      w.int(a)
      w.int(b)
      w.bool(c)
    }
    MoeSigmoidAndCorrectionBias(a) => {
      w.byte(21)
      w.int(a)
    }
    ExpertScoreTransform(a) => {
      w.byte(22)
      encode_score(w, a)
    }
    MoeGroupedNoAuxiliaryLossTopK(a, b, c, d, e, f, g) => {
      w.byte(23)
      w.int(a)
      w.int(b)
      w.int(c)
      w.int(d)
      w.int(e)
      w.bool(f)
      w.double(g)
    }
    RoutedExpertFeedForward(a, b, c) => {
      w.byte(24)
      w.int(a)
      w.int(b)
      encode_activation(w, c)
    }
    SharedExpertFeedForward(a, b, c) => {
      w.byte(25)
      w.int(a)
      w.int(b)
      encode_activation(w, c)
    }
    RoutedSharedExpertCombine(a, b, c) => {
      w.byte(26)
      w.int(a)
      w.bool(b)
      w.double(c)
    }
    MultiTokenPredictionInputProjection => w.byte(27)
    AuxiliaryDecoderStage(a) => {
      w.byte(28)
      w.int(a)
    }
    AuxiliaryOutputHead => w.byte(29)
    VisionPatchProjection(a, b, c, d) => {
      w.byte(30)
      w.int(a)
      w.int(b)
      w.int(c)
      w.int(d)
    }
    VisionRmsNorm(a) => {
      w.byte(31)
      w.int(a)
    }
    VisionQkvProjection(a, b) => {
      w.byte(32)
      w.int(a)
      w.int(b)
    }
    VisionAttention(a, b) => {
      w.byte(33)
      w.int(a)
      w.int(b)
    }
    VisionOutputProjection(a) => {
      w.byte(34)
      w.int(a)
    }
    VisionSwiGlu(a, b) => {
      w.byte(35)
      w.int(a)
      w.int(b)
    }
    VisionSpatialMerge(a, b, c) => {
      w.byte(36)
      w.int(a)
      w.int(b)
      w.int(c)
    }
    VisionDecoderProjection(a, b) => {
      w.byte(37)
      w.int(a)
      w.int(b)
    }
    VisionPostProjectionNorm(a) => {
      w.byte(38)
      w.int(a)
    }
    FinalRmsNorm(a, b) => {
      w.byte(39)
      w.int(a)
      w.double(b)
    }
    LanguageModelHead(a, b) => {
      w.byte(40)
      w.int(a)
      w.int(b)
    }
    DsaQueryAProjection(a, b) => {
      w.byte(41)
      w.int(a)
      w.int(b)
    }
    DsaKeyValueAProjection(a, b, c) => {
      w.byte(42)
      w.int(a)
      w.int(b)
      w.int(c)
    }
    DsaQueryARmsNorm(a, b) => {
      w.byte(43)
      w.int(a)
      w.double(b)
    }
    DsaQueryBProjection(a, b, c) => {
      w.byte(44)
      w.int(a)
      w.int(b)
      w.int(c)
    }
    KdaDecayControl(a, b, c) => {
      w.byte(45)
      w.int(a)
      w.int(b)
      w.double(c)
    }
    KdaOutputGateProjection(a, b, c) => {
      w.byte(46)
      w.int(a)
      w.int(b)
      w.int(c)
    }
    KdaGatedRmsNorm(a, b, c) => {
      w.byte(47)
      w.int(a)
      w.int(b)
      w.double(c)
    }
    MoeDecoderResidualAdd(a) => {
      w.byte(48)
      w.int(a)
    }
    DsaKeyValueAttentionAssembly(a, b, c, d) => {
      w.byte(49)
      w.int(a)
      w.int(b)
      w.int(c)
      w.int(d)
    }
  }
}

///|
fn advanced_capability_tag(
  value : @advanced_decoder_kernel_requirements.AdvancedDecoderAotCapability,
) -> Byte {
  match value {
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
fn encode_capability(
  w : Writer,
  value : GlmHybridAotCapability,
) -> Unit raise GlmHybridKernelRequirementsError {
  match value {
    Advanced(a) => {
      w.byte(1)
      w.byte(advanced_capability_tag(a))
    }
    DecoderRmsNormAot => w.byte(2)
    GlmRotaryPositionAot => w.byte(3)
    DsaIndexComputeAot => w.byte(4)
    DsaIndexPoolingAot => w.byte(5)
    DsaIndexReuseAot => w.byte(6)
    DsaSparseAttentionAot => w.byte(7)
    AttentionOutputProjectionAot => w.byte(8)
    KdaProjectionAot => w.byte(9)
    KdaShortConvolutionAot => w.byte(10)
    KdaRecurrentUpdateAot => w.byte(11)
    KdaOutputProjectionAot => w.byte(12)
    HyperConnectionFunctionAot => w.byte(13)
    DenseSwiGluAot => w.byte(14)
    MoeRouterProjectionAot => w.byte(15)
    MoeSigmoidAndCorrectionBiasAot => w.byte(16)
    VisionPatchProjectionAot => w.byte(17)
    VisionRmsNormAot => w.byte(18)
    VisionQkvProjectionAot => w.byte(19)
    VisionAttentionAot => w.byte(20)
    VisionOutputProjectionAot => w.byte(21)
    VisionSwiGluAot => w.byte(22)
    VisionSpatialMergeAot => w.byte(23)
    VisionDecoderProjectionAot => w.byte(24)
    VisionPostProjectionNormAot => w.byte(25)
    DsaQueryAProjectionAot => w.byte(26)
    DsaKeyValueAProjectionAot => w.byte(27)
    DsaQueryARmsNormAot => w.byte(28)
    DsaQueryBProjectionAot => w.byte(29)
    KdaDecayControlAot => w.byte(30)
    KdaOutputGateProjectionAot => w.byte(31)
    KdaGatedRmsNormAot => w.byte(32)
    MoeGroupedRoutingAot => w.byte(33)
    MoeDecoderResidualAddAot => w.byte(34)
    DsaKeyValueAttentionAssemblyAot => w.byte(35)
  }
}

///|
fn canonical_digest(
  identity : @spec.ModelIdentity,
  entries : ArrayView[GlmHybridKernelRequirement],
) -> GlmHybridKernelRequirementsDigest raise GlmHybridKernelRequirementsError {
  let w = Writer::new()
  w.text("lunaflux/glm-hybrid-kernel-requirements/v1")
  w.text(identity.content().as_hex())
  w.text(identity.plan().as_hex())
  w.int(entries.length())
  for entry in entries {
    encode_operation(w, entry.operation)
    encode_capability(w, entry.capability)
  }
  { hex: w.hash.finalize_hex(), }
}
