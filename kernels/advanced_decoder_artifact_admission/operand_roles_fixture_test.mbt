///|
fn artifact_test_roles(
  operation : @advanced_decoder_kernel_requirements.AdvancedDecoderSemanticOperation,
) -> Array[AdvancedDecoderLaunchOperandRole] {
  match operation {
    TokenEmbedding =>
      [StepCountsInput, TokenIdsInput, ModelWeights, HiddenOutput]
    QueryLowRankAProjection(_) =>
      [
        StepCountsInput,
        HiddenInput,
        QueryAWeightCodesInput,
        QueryAWeightScalesInput,
        QueryLowRankAOutput,
      ]
    QueryLowRankBProjection(_, _, _) => [HiddenInput]
    KeyValueProjection(_, _, _) =>
      [
        StepCountsInput,
        HiddenInput,
        KeyValueWeightCodesInput,
        KeyValueWeightScalesInput,
        KeyValueProjectionOutput,
      ]
    AttentionOutputAProjectionV1(_, _, _, _) =>
      [
        StepCountsInput,
        AttentionOutputProjectionInput,
        AttentionOutputAWeightsInput,
        AttentionOutputRankOutput,
      ]
    AttentionOutputBProjectionV1(_, _, _) =>
      [
        StepCountsInput,
        AttentionOutputRankInput,
        AttentionOutputBWeightCodesInput,
        AttentionOutputBWeightScalesInput,
        HiddenOutput,
      ]
    FinalRmsNorm(_) =>
      [StepCountsInput, HiddenInput, ModelWeights, HiddenOutput]
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
    LanguageModelHead =>
      [StepCountsInput, HiddenInput, ModelWeights, LogitsOutput]
  }
}
