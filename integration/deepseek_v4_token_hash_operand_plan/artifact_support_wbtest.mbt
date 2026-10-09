///|
fn artifact_binding_test_module_bytes() -> Bytes {
  b"unit-test-token-hash-cubin"
}

///|
fn artifact_binding_test_module_digest() -> @catalog.AotArtifactDigest {
  let bytes = artifact_binding_test_module_bytes()
  try! @catalog.AotArtifactDigest::from_sha256(
    @crypto.bytes_to_hex_string(@crypto.sha256(bytes)),
  )
}

///|
fn artifact_binding_test_entry(ordinal : Int) -> @catalog.AotKernelEntryPoint {
  @catalog.AotKernelEntryPoint::new(
    family=@catalog.AotKernelFamily::new(
      module_digest=artifact_binding_test_module_digest(),
      family_id=try! @catalog.AotKernelFamilyId::from_stable_int(ordinal + 1),
    ),
    entry_point_id=try! @catalog.AotKernelEntryPointId::from_stable_int(1),
  )
}

///|
fn artifact_binding_test_dimensions(
  requirements : @advanced_decoder_kernel_requirements.AdvancedDecoderKernelRequirements,
  operation : @advanced_decoder_kernel_requirements.AdvancedDecoderSemanticOperation,
) -> @launch_contract.AotLaunchDimensions {
  match operation {
    HyperConnectionBlockControlV1(_, _, _, _) =>
      try! @launch_contract.AotLaunchDimensions::new(
        grid_x=2,
        grid_y=1,
        grid_z=1,
        block_x=1,
        block_y=1,
        block_z=1,
        shared_memory_bytes=0,
      )
    HyperConnectionPreReduceV1(_) =>
      try! @launch_contract.AotLaunchDimensions::new(
        grid_x=2,
        grid_y=1,
        grid_z=1,
        block_x=256,
        block_y=1,
        block_z=1,
        shared_memory_bytes=0,
      )
    HyperConnectionPostCombineV1(_) | HyperConnectionHeadReduceV1(_, _, _) =>
      try! @launch_contract.AotLaunchDimensions::new(
        grid_x=2,
        grid_y=1,
        grid_z=1,
        block_x=256,
        block_y=1,
        block_z=1,
        shared_memory_bytes=0,
      )
    QueryKeyRmsNorm(_, _, _) =>
      try! @launch_contract.AotLaunchDimensions::new(
        grid_x=2,
        grid_y=2,
        grid_z=1,
        block_x=256,
        block_y=1,
        block_z=1,
        shared_memory_bytes=1024,
      )
    RotaryPositionEncoding(_, _, _, _, _, _) =>
      try! @launch_contract.AotLaunchDimensions::new(
        grid_x=2,
        grid_y=requirements.geometry().attention_head_count() +
          requirements.geometry().kv_head_count(),
        grid_z=1,
        block_x=32,
        block_y=1,
        block_z=1,
        shared_memory_bytes=0,
      )
    TokenHashRouting(_, _, _, _) =>
      try! @launch_contract.AotLaunchDimensions::new(
        grid_x=2,
        grid_y=1,
        grid_z=1,
        block_x=256,
        block_y=1,
        block_z=1,
        shared_memory_bytes=0,
      )
    NoAuxiliaryLossTopK(_, _) | SelectedExpertWeightFinalize(_, _, _, _) =>
      try! @launch_contract.AotLaunchDimensions::new(
        grid_x=2,
        grid_y=1,
        grid_z=1,
        block_x=1,
        block_y=1,
        block_z=1,
        shared_memory_bytes=0,
      )
    _ =>
      try! @launch_contract.AotLaunchDimensions::new(
        grid_x=2,
        grid_y=1,
        grid_z=1,
        block_x=32,
        block_y=1,
        block_z=1,
        shared_memory_bytes=0,
      )
  }
}

///|
fn artifact_binding_test_roles(
  operation : @advanced_decoder_kernel_requirements.AdvancedDecoderSemanticOperation,
) -> Array[
  @advanced_decoder_artifact_admission.AdvancedDecoderLaunchOperandRole,
] {
  match operation {
    TokenEmbedding =>
      [StepCountsInput, TokenIdsInput, ModelWeights, HiddenOutput]
    QueryLowRankAProjection(_) | QueryLowRankBProjection(_, _, _) =>
      [HiddenInput]
    KeyValueProjection(_, _, _) => [HiddenInput, ModelWeights, HiddenOutput]
    AttentionOutputAProjectionV1(_, _, _, _) =>
      [
        StepCountsInput,
        AttentionOutputProjectionInput,
        AttentionOutputAWeightsInput,
        AttentionOutputRankOutput,
      ]
    AttentionOutputBProjectionV1(_, _, _) => [HiddenInput]
    FinalRmsNorm(_) =>
      [StepCountsInput, HiddenInput, ModelWeights, HiddenOutput]
    LanguageModelHead =>
      [StepCountsInput, HiddenInput, ModelWeights, LogitsOutput]
    QueryKeyRmsNorm(_, _, _) =>
      [
        StepCountsInput,
        QueryInput,
        QueryNormWeights,
        QueryOutput,
        KeyValueInput,
        KeyValueNormWeights,
        KeyValueOutput,
      ]
    RotaryPositionEncoding(_, _, _, _, _, _) =>
      [
        StepCountsInput,
        PositionInput,
        QueryInput,
        QueryOutput,
        KeyValueInput,
        KeyValueOutput,
      ]
    HyperConnectionBlockControlV1(_, _, _, _) =>
      [
        StepCountsInput,
        HyperConnectionStateInput,
        HyperConnectionFunctionWeightsInput,
        HyperConnectionBaseInput,
        HyperConnectionScaleInput,
        HyperConnectionPreOutput,
        HyperConnectionPostOutput,
        HyperConnectionCombinationOutput,
      ]
    HyperConnectionPreReduceV1(_) =>
      [
        StepCountsInput,
        HyperConnectionStateInput,
        HyperConnectionPreInput,
        HyperConnectionReducedOutput,
      ]
    HyperConnectionPostCombineV1(_) =>
      [
        StepCountsInput,
        HyperConnectionBranchInput,
        HyperConnectionStateInput,
        HyperConnectionPostInput,
        HyperConnectionCombinationInput,
        HyperConnectionCombinedOutput,
      ]
    HyperConnectionHeadReduceV1(_, _, _) =>
      [
        StepCountsInput,
        HyperConnectionStateInput,
        HyperConnectionFunctionWeightsInput,
        HyperConnectionBaseInput,
        HyperConnectionScaleInput,
        HyperConnectionReducedOutput,
      ]
    SlidingWindowCausalAttention(_) | IndexedCompressedAttention(_) =>
      [HiddenInput, ModelWeights, HiddenOutput, Workspace]
    NonOverlappingCompressedKeyValuePrepareV1(_, _) =>
      [
        StepCountsInput,
        PreparedKeyValueModesInput,
        PreparedKeyValueSequenceLengthsInput,
        UncompressedKeyValueInput,
        CompressedKeyValueInput,
        PreparedKeyValueLengthsOutput,
        PreparedKeyValueOutput,
      ]
    AttentionOutputInverseRotaryV1(_, _, _, _, _) =>
      [StepCountsInput, PositionInput, AttentionOutputRotaryInOut]
    LearnedCompressedIndex(_, _, _, _) =>
      [HiddenInput, ModelWeights, RoutingIndicesOutput, Workspace]
    DeterministicCompressedIndex(_) =>
      [
        StepCountsInput,
        CompressedQueryPositionsInput,
        CompressedCacheOffsetsInput,
        CompressedIndexCountsOutput,
        CompressedIndicesOutput,
      ]
    DeterministicWindowCompressedIndexJoinV1(_, _) =>
      [
        StepCountsInput,
        WindowQueryPositionsInput,
        WindowSequenceLengthsInput,
        WindowQueryModesInput,
        CompressedIndexCountsInput,
        CompressedIndicesInput,
        JoinedIndexCountsOutput,
        JoinedIndicesOutput,
      ]
    SelectedSparseAttentionV1(_, _) =>
      [
        StepCountsInput,
        SparseAttentionQueryInput,
        SparseAttentionQueryBatchIdsInput,
        SparseAttentionKeyValueInput,
        SparseAttentionCacheLengthsInput,
        SparseAttentionSinkInput,
        SparseAttentionSelectedIndicesInput,
        SparseAttentionOutput,
      ]
    ExpertScoreTransform(_, _) =>
      [StepCountsInput, RoutingScoresInput, RoutingScoresOutput]
    TokenHashRouting(_, _, _, _) =>
      [
        StepCountsInput,
        TokenIdsInput,
        TokenHashTableInput,
        RoutingIndicesOutput,
      ]
    NoAuxiliaryLossTopK(_, _) =>
      [
        StepCountsInput,
        RoutingScoresInput,
        RoutingSelectionBiasInput,
        RoutingIndicesOutput,
      ]
    SelectedExpertWeightFinalize(_, _, _, _) =>
      [
        StepCountsInput,
        RoutingScoresInput,
        RoutingIndicesInput,
        RoutingWeightsOutput,
      ]
    RoutedExpertFeedForward(_, _, _) =>
      [HiddenInput, RoutingIndicesInput, ModelWeights, ExpertOutput]
    SharedExpertFeedForward(_, _, _) =>
      [HiddenInput, ModelWeights, ExpertOutput]
    RoutedSharedExpertCombine(_, _, _) =>
      [ExpertInput, RoutingScoresInput, HiddenOutput]
    MultiTokenPredictionInputProjection =>
      [HiddenInput, ModelWeights, AuxiliaryOutput]
    DsparkTargetLayerRead(_) => [HiddenInput, AuxiliaryOutput]
    DsparkMainProjection
    | DsparkNoiseEmbedding(_, _)
    | AuxiliaryDecoderStage(_)
    | AuxiliaryOutputHead
    | DsparkMarkovAdjustment(_)
    | DsparkConfidence => [AuxiliaryInput, ModelWeights, AuxiliaryOutput]
  }
}

///|
fn artifact_binding_test_operands(
  requirements : @advanced_decoder_kernel_requirements.AdvancedDecoderKernelRequirements,
  operation : @advanced_decoder_kernel_requirements.AdvancedDecoderSemanticOperation,
) -> Array[@advanced_decoder_artifact_admission.AdvancedDecoderLaunchOperand] {
  let operands : Array[
    @advanced_decoder_artifact_admission.AdvancedDecoderLaunchOperand,
  ] = []
  for index, role in artifact_binding_test_roles(operation) {
    let (byte_count, alignment) = match operation {
      LanguageModelHead if index == 0 => (20L, 4L)
      ExpertScoreTransform(_, experts) =>
        if index == 0 {
          (20L, 4L)
        } else {
          (experts.to_int64() * 8L, 4L)
        }
      QueryKeyRmsNorm(_, query_width, key_value_width) =>
        match index {
          0 => (20L, 4L)
          1 | 3 => (query_width.to_int64() * 4L, 16L)
          2 => (query_width.to_int64() * 2L, 16L)
          4 | 6 => (key_value_width.to_int64() * 4L, 16L)
          _ => (key_value_width.to_int64() * 2L, 16L)
        }
      RotaryPositionEncoding(_, _, _, _, _, _) => {
        let geometry = requirements.geometry()
        let query_bytes = geometry.attention_head_count().to_int64() * 256L
        let key_value_bytes = geometry.kv_head_count().to_int64() * 256L
        match index {
          0 => (20L, 4L)
          1 => (8L, 4L)
          2 | 3 => (query_bytes, 16L)
          _ => (key_value_bytes, 16L)
        }
      }
      HyperConnectionBlockControlV1(width, _, _, _) => {
        let rows = 2L
        let stream = width.to_int64()
        let hidden = requirements.geometry().hidden_size().to_int64()
        match index {
          0 => (20L, 4L)
          1 => (rows * stream * hidden * 2L, 16L)
          2 => (6L * stream * stream * hidden * 4L, 16L)
          3 => (6L * stream * 4L, 16L)
          4 => (12L, 4L)
          5 | 6 => (rows * stream * 4L, 16L)
          _ => (rows * stream * stream * 4L, 16L)
        }
      }
      HyperConnectionPreReduceV1(width) => {
        let rows = 2L
        let stream = width.to_int64()
        let hidden = requirements.geometry().hidden_size().to_int64()
        match index {
          0 => (20L, 4L)
          1 => (rows * stream * hidden * 2L, 16L)
          2 => (rows * stream * 4L, 16L)
          _ => (rows * hidden * 2L, 16L)
        }
      }
      HyperConnectionPostCombineV1(width) => {
        let rows = 2L
        let stream = width.to_int64()
        let hidden = requirements.geometry().hidden_size().to_int64()
        match index {
          0 => (20L, 4L)
          1 => (rows * hidden * 2L, 16L)
          2 => (rows * stream * hidden * 2L, 16L)
          3 => (rows * stream * 4L, 16L)
          4 => (rows * stream * stream * 4L, 16L)
          _ => (rows * stream * hidden * 2L, 16L)
        }
      }
      HyperConnectionHeadReduceV1(width, _, _) => {
        let rows = 2L
        let stream = width.to_int64()
        let hidden = requirements.geometry().hidden_size().to_int64()
        match index {
          0 => (20L, 4L)
          1 => (rows * stream * hidden * 2L, 16L)
          2 => (stream * stream * hidden * 4L, 16L)
          3 => (stream * 4L, 16L)
          4 => (4L, 4L)
          _ => (rows * hidden * 2L, 16L)
        }
      }
      TokenHashRouting(_, vocabulary, selected, _) =>
        match index {
          0 => (20L, 4L)
          1 => (8L, 4L)
          2 => (vocabulary.to_int64() * selected.to_int64() * 4L, 4L)
          _ => (selected.to_int64() * 8L, 4L)
        }
      NoAuxiliaryLossTopK(experts, selected) =>
        match index {
          0 => (20L, 4L)
          1 => (experts.to_int64() * 8L, 4L)
          2 => (experts.to_int64() * 4L, 4L)
          _ => (selected.to_int64() * 8L, 4L)
        }
      SelectedExpertWeightFinalize(experts, selected, _, _) =>
        match index {
          0 => (20L, 4L)
          1 => (experts.to_int64() * 8L, 4L)
          _ => (selected.to_int64() * 8L, 4L)
        }
      _ => (64L, 16L)
    }
    operands.push(
      try! @advanced_decoder_artifact_admission.AdvancedDecoderLaunchOperand::new(
        role~,
        byte_count~,
        alignment~,
      ),
    )
  }
  operands
}

///|
fn artifact_binding_test_admission(
  requirements : @advanced_decoder_kernel_requirements.AdvancedDecoderKernelRequirements,
  candidate : @advanced_decoder_cuda_aot.AdvancedDecoderRoutingCudaCandidate,
  target? : @catalog.DeviceTarget = candidate.target(),
  hash_symbol? : String = candidate.function_symbol(),
  hash_candidate? : @advanced_decoder_cuda_aot.AdvancedDecoderRoutingCudaCandidate = candidate,
) -> @advanced_decoder_artifact_admission.AdvancedDecoderArtifactAdmission raise @advanced_decoder_artifact_admission.AdvancedDecoderArtifactAdmissionError {
  let query_a = try! @advanced_decoder_cuda_aot.lower_query_a_projection_cuda_aot_candidate(
    requirements,
    requirement_ordinal=1,
    maximum_tokens=2,
    target~,
    compiler=operand_plan_test_compiler(),
    limits=operand_plan_test_limits(),
  )
  let key_value = artifact_binding_test_key_value_candidate(
    requirements, target,
  )
  let launches : Array[
    @advanced_decoder_artifact_admission.AdvancedDecoderLaunchEvidenceInput,
  ] = []
  let entries : Array[@artifact.KernelEntryPointInput] = []
  for ordinal, requirement in requirements.requirements() {
    let operation = requirement.operation()
    let is_hash = ordinal == candidate.requirement_ordinal()
    let is_query_a = ordinal == query_a.requirement_ordinal()
    let is_key_value = ordinal == key_value.requirement_ordinal()
    launches.push(
      try! @advanced_decoder_artifact_admission.AdvancedDecoderLaunchEvidenceInput::new(
        requirements_digest=requirements.digest(),
        source_identity=requirements.source_identity(),
        requirement_ordinal=ordinal,
        operation~,
        capability=requirement.capability(),
        target~,
        entry_point=artifact_binding_test_entry(ordinal),
        module_digest=artifact_binding_test_module_digest(),
        dimensions=if is_hash {
          hash_candidate.dimensions()
        } else if is_query_a {
          query_a.dimensions()
        } else if is_key_value {
          key_value.dimensions()
        } else {
          artifact_binding_test_dimensions(requirements, operation)
        },
        operands=if is_hash {
          hash_candidate.operands()
        } else if is_query_a {
          query_a.operands()
        } else if is_key_value {
          key_value.operands()
        } else {
          artifact_binding_test_operands(requirements, operation)
        },
      ),
    )
    entries.push(
      @artifact.KernelEntryPointInput::new(
        entry_point=artifact_binding_test_entry(ordinal),
        function_symbol=if is_hash {
          hash_symbol
        } else if is_query_a {
          query_a.function_symbol()
        } else if is_key_value {
          key_value.function_symbol()
        } else {
          "advanced_\{ordinal}"
        },
      ),
    )
  }
  @advanced_decoder_artifact_admission.admit(
    requirements,
    target~,
    module_inputs=[
      @advanced_decoder_artifact_admission.AdvancedDecoderModuleInput::new(
        digest=artifact_binding_test_module_digest(),
        module_bytes=artifact_binding_test_module_bytes(),
      ),
    ],
    entry_point_inputs=entries,
    launch_inputs=launches,
    limits=try! @advanced_decoder_artifact_admission.AdvancedDecoderArtifactAdmissionLimits::new(
      maximum_requirements=64,
      maximum_modules=4,
      maximum_entry_points=64,
      maximum_module_bytes=1024L,
      maximum_total_module_bytes=4096L,
      maximum_symbol_bytes=128,
      maximum_operands_per_launch=8,
      maximum_canonical_bytes=262144,
    ),
  )
}
