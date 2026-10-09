///|
priv struct RequirementCanonicalWriter {
  hasher : @canonical_sha256.CanonicalSha256
  mut byte_count : Int
}

///|
fn RequirementCanonicalWriter::new() -> Self {
  { hasher: @canonical_sha256.CanonicalSha256::new(), byte_count: 0, }
}

///|
fn RequirementCanonicalWriter::reserve(
  self : Self,
  count : Int,
) -> Unit raise AdvancedDecoderKernelRequirementsError {
  if count < 0 || self.byte_count > 65536 - count {
    raise RequirementsEncodingTooLarge
  }
  self.byte_count += count
}

///|
fn RequirementCanonicalWriter::byte(
  self : Self,
  value : Byte,
) -> Unit raise AdvancedDecoderKernelRequirementsError {
  self.reserve(1)
  if !self.hasher.update_byte(value) {
    abort("finalized advanced decoder requirement hash")
  }
}

///|
fn RequirementCanonicalWriter::int(
  self : Self,
  value : Int,
) -> Unit raise AdvancedDecoderKernelRequirementsError {
  self.reserve(8)
  if !self.hasher.update_u64_le(Int64::from_int(value).reinterpret_as_uint64()) {
    abort("finalized advanced decoder requirement hash")
  }
}

///|
fn RequirementCanonicalWriter::double(
  self : Self,
  value : Double,
) -> Unit raise AdvancedDecoderKernelRequirementsError {
  self.reserve(8)
  if !self.hasher.update_u64_le(value.reinterpret_as_uint64()) {
    abort("finalized advanced decoder requirement hash")
  }
}

///|
fn RequirementCanonicalWriter::text(
  self : Self,
  value : String,
) -> Unit raise AdvancedDecoderKernelRequirementsError {
  let bytes = @utf8.encode(value)
  self.int(bytes.length())
  self.reserve(bytes.length())
  if !self.hasher.update_bytes(bytes) {
    abort("finalized advanced decoder requirement hash")
  }
}

///|
fn encode_score_function(
  writer : RequirementCanonicalWriter,
  value : @advanced_decoder_execution_plan.ExpertScoreFunction,
) -> Unit raise AdvancedDecoderKernelRequirementsError {
  writer.byte(
    match value {
      SqrtSoftplus => 1
      Sigmoid => 2
    },
  )
}

///|
fn encode_activation(
  writer : RequirementCanonicalWriter,
  value : @advanced_decoder_execution_plan.ExpertActivation,
) -> Unit raise AdvancedDecoderKernelRequirementsError {
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
  writer : RequirementCanonicalWriter,
  operation : AdvancedDecoderSemanticOperation,
) -> Unit raise AdvancedDecoderKernelRequirementsError {
  match operation {
    TokenEmbedding => writer.byte(1)
    QueryLowRankAProjection(rank) => {
      writer.byte(2)
      writer.int(rank)
    }
    QueryLowRankBProjection(rank, heads, head_dimension) => {
      writer.byte(33)
      writer.int(rank)
      writer.int(heads)
      writer.int(head_dimension)
    }
    KeyValueProjection(heads, head_dimension, rope_dimension) => {
      writer.byte(3)
      writer.int(heads)
      writer.int(head_dimension)
      writer.int(rope_dimension)
    }
    QueryKeyRmsNorm(epsilon, query_width, key_value_width) => {
      writer.byte(4)
      writer.double(epsilon)
      writer.int(query_width)
      writer.int(key_value_width)
    }
    RotaryPositionEncoding(
      theta,
      compressed_theta,
      original_positions,
      factor,
      beta_fast,
      beta_slow
    ) => {
      writer.byte(5)
      writer.double(theta)
      writer.double(compressed_theta)
      writer.int(original_positions)
      writer.double(factor)
      writer.double(beta_fast)
      writer.double(beta_slow)
    }
    HyperConnectionBlockControlV1(
      width,
      iterations,
      control_epsilon,
      normalization_epsilon
    ) => {
      writer.byte(6)
      writer.int(width)
      writer.int(iterations)
      writer.double(control_epsilon)
      writer.double(normalization_epsilon)
    }
    HyperConnectionPreReduceV1(width) => {
      writer.byte(7)
      writer.int(width)
    }
    HyperConnectionPostCombineV1(width) => {
      writer.byte(31)
      writer.int(width)
    }
    HyperConnectionHeadReduceV1(width, control_epsilon, normalization_epsilon) => {
      writer.byte(32)
      writer.int(width)
      writer.double(control_epsilon)
      writer.double(normalization_epsilon)
    }
    SlidingWindowCausalAttention(window) => {
      writer.byte(8)
      writer.int(window)
    }
    LearnedCompressedIndex(ratio, heads, dimension, top_k) => {
      writer.byte(9)
      writer.int(ratio)
      writer.int(heads)
      writer.int(dimension)
      writer.int(top_k)
    }
    IndexedCompressedAttention(ratio) => {
      writer.byte(10)
      writer.int(ratio)
    }
    DeterministicCompressedIndex(ratio) => {
      writer.byte(11)
      writer.int(ratio)
    }
    DeterministicWindowCompressedIndexJoinV1(window, ratio) => {
      writer.byte(34)
      writer.int(window)
      writer.int(ratio)
    }
    NonOverlappingCompressedKeyValuePrepareV1(window, ratio) => {
      writer.byte(35)
      writer.int(window)
      writer.int(ratio)
    }
    SelectedSparseAttentionV1(window, ratio) => {
      writer.byte(36)
      writer.int(window)
      writer.int(ratio)
    }
    AttentionOutputInverseRotaryV1(theta, original, factor, fast, slow) => {
      writer.byte(37)
      writer.double(theta)
      writer.int(original)
      writer.double(factor)
      writer.double(fast)
      writer.double(slow)
    }
    AttentionOutputAProjectionV1(rank, groups, heads, head_dimension) => {
      writer.byte(38)
      writer.int(rank)
      writer.int(groups)
      writer.int(heads)
      writer.int(head_dimension)
    }
    AttentionOutputBProjectionV1(rank, groups, hidden) => {
      writer.byte(39)
      writer.int(rank)
      writer.int(groups)
      writer.int(hidden)
    }
    ExpertScoreTransform(function, routed_expert_count) => {
      writer.byte(14)
      encode_score_function(writer, function)
      writer.int(routed_expert_count)
    }
    TokenHashRouting(
      layer_count,
      vocabulary_size,
      experts_per_token,
      routed_expert_count
    ) => {
      writer.byte(15)
      writer.int(layer_count)
      writer.int(vocabulary_size)
      writer.int(experts_per_token)
      writer.int(routed_expert_count)
    }
    NoAuxiliaryLossTopK(routed_expert_count, top_k) => {
      writer.byte(16)
      writer.int(routed_expert_count)
      writer.int(top_k)
    }
    SelectedExpertWeightFinalize(routed_expert_count, top_k, normalize, scaling) => {
      writer.byte(30)
      writer.int(routed_expert_count)
      writer.int(top_k)
      writer.byte(if normalize { 1 } else { 0 })
      writer.double(scaling)
    }
    RoutedExpertFeedForward(count, intermediate, activation) => {
      writer.byte(17)
      writer.int(count)
      writer.int(intermediate)
      encode_activation(writer, activation)
    }
    SharedExpertFeedForward(count, intermediate, activation) => {
      writer.byte(18)
      writer.int(count)
      writer.int(intermediate)
      encode_activation(writer, activation)
    }
    RoutedSharedExpertCombine(top_k, normalize, scaling) => {
      writer.byte(19)
      writer.int(top_k)
      writer.byte(if normalize { 1 } else { 0 })
      writer.double(scaling)
    }
    MultiTokenPredictionInputProjection => writer.byte(20)
    DsparkTargetLayerRead(layer) => {
      writer.byte(21)
      writer.int(layer)
    }
    DsparkMainProjection => writer.byte(22)
    DsparkNoiseEmbedding(block_size, token_id) => {
      writer.byte(23)
      writer.int(block_size)
      writer.int(token_id)
    }
    AuxiliaryDecoderStage(stage) => {
      writer.byte(24)
      writer.int(stage)
    }
    AuxiliaryOutputHead => writer.byte(25)
    DsparkMarkovAdjustment(rank) => {
      writer.byte(26)
      writer.int(rank)
    }
    DsparkConfidence => writer.byte(27)
    FinalRmsNorm(epsilon) => {
      writer.byte(28)
      writer.double(epsilon)
    }
    LanguageModelHead => writer.byte(29)
  }
}

///|
fn capability_tag(capability : AdvancedDecoderAotCapability) -> Byte {
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
fn canonical_requirements_digest(
  identity : @spec.ModelIdentity,
  entries : ArrayView[AdvancedDecoderKernelRequirement],
) -> AdvancedDecoderKernelRequirementsDigest raise AdvancedDecoderKernelRequirementsError {
  let writer = RequirementCanonicalWriter::new()
  writer.text("lunaflux/advanced-decoder-kernel-requirements/v2")
  writer.text(identity.content().as_hex())
  writer.text(identity.plan().as_hex())
  writer.int(entries.length())
  for entry in entries {
    encode_operation(writer, entry.operation)
    writer.byte(capability_tag(entry.capability))
  }
  { hex: writer.hasher.finalize_hex(), }
}
