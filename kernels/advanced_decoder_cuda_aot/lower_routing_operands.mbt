///|
fn routing_operand_text(
  operands : ArrayView[
    @advanced_decoder_artifact_admission.AdvancedDecoderLaunchOperand,
  ],
) -> String {
  let text = StringBuilder()
  for index, item in operands {
    if index > 0 {
      text.write_char(',')
    }
    let role = match item.role() {
      StepCountsInput => "step-counts"
      TokenIdsInput => "token-ids"
      PositionInput => "positions"
      HiddenInput => "hidden-input"
      HiddenOutput => "hidden-output"
      ModelWeights => "model-weights"
      HyperConnectionStateInput => "hyper-state-input"
      HyperConnectionStateOutput => "hyper-state-output"
      HyperConnectionFunctionWeightsInput => "hyper-function-weights-input"
      HyperConnectionBaseInput => "hyper-base-input"
      HyperConnectionScaleInput => "hyper-scale-input"
      HyperConnectionPreOutput => "hyper-pre-output"
      HyperConnectionPostOutput => "hyper-post-output"
      HyperConnectionCombinationOutput => "hyper-combination-output"
      HyperConnectionPreInput => "hyper-pre-input"
      HyperConnectionReducedOutput => "hyper-reduced-output"
      HyperConnectionBranchInput => "hyper-branch-input"
      HyperConnectionPostInput => "hyper-post-input"
      HyperConnectionCombinationInput => "hyper-combination-input"
      HyperConnectionCombinedOutput => "hyper-combined-output"
      RoutingScoresInput => "routing-scores-input"
      RoutingScoresOutput => "routing-scores-output"
      RoutingIndicesInput => "routing-indices-input"
      RoutingIndicesOutput => "routing-indices-output"
      CompressedQueryPositionsInput => "compressed-query-positions-input"
      CompressedCacheOffsetsInput => "compressed-cache-offsets-input"
      CompressedIndexCountsOutput => "compressed-index-counts-output"
      CompressedIndicesOutput => "compressed-indices-output"
      SparseAttentionQueryInput => "sparse-attention-query-input"
      SparseAttentionQueryBatchIdsInput =>
        "sparse-attention-query-batch-ids-input"
      SparseAttentionKeyValueInput => "sparse-attention-key-value-input"
      SparseAttentionCacheLengthsInput => "sparse-attention-cache-lengths-input"
      SparseAttentionSinkInput => "sparse-attention-sink-input"
      SparseAttentionSelectedIndicesInput =>
        "sparse-attention-selected-indices-input"
      SparseAttentionOutput => "sparse-attention-output"
      WindowQueryPositionsInput => "window-query-positions-input"
      WindowSequenceLengthsInput => "window-sequence-lengths-input"
      WindowQueryModesInput => "window-query-modes-input"
      CompressedIndexCountsInput => "compressed-index-counts-input"
      CompressedIndicesInput => "compressed-indices-input"
      JoinedIndexCountsOutput => "joined-index-counts-output"
      JoinedIndicesOutput => "joined-indices-output"
      PreparedKeyValueModesInput => "prepared-kv-modes-input"
      PreparedKeyValueSequenceLengthsInput => "prepared-kv-lengths-input"
      UncompressedKeyValueInput => "uncompressed-kv-input"
      CompressedKeyValueInput => "compressed-kv-input"
      PreparedKeyValueLengthsOutput => "prepared-kv-lengths-output"
      PreparedKeyValueOutput => "prepared-kv-output"
      AttentionOutputRotaryInOut => "attention-output-rotary-inout"
      AttentionOutputProjectionInput => "attention-output-projection-input"
      AttentionOutputAWeightsInput => "attention-output-a-weights-input"
      AttentionOutputRankOutput => "attention-output-rank-output"
      AttentionOutputRankInput => "attention-output-rank-input"
      AttentionOutputBWeightCodesInput =>
        "attention-output-b-weight-codes-input"
      AttentionOutputBWeightScalesInput =>
        "attention-output-b-weight-scales-input"
      QueryAWeightCodesInput => "query-a-weight-codes-input"
      QueryAWeightScalesInput => "query-a-weight-scales-input"
      QueryLowRankAOutput => "query-low-rank-a-output"
      KeyValueWeightCodesInput => "key-value-weight-codes-input"
      KeyValueWeightScalesInput => "key-value-weight-scales-input"
      KeyValueProjectionOutput => "key-value-projection-output"
      QueryInput => "query-input"
      QueryNormWeights => "query-norm-weights"
      QueryOutput => "query-output"
      KeyValueInput => "key-value-input"
      KeyValueNormWeights => "key-value-norm-weights"
      KeyValueOutput => "key-value-output"
      TokenHashTableInput => "token-hash-table-input"
      RoutingSelectionBiasInput => "routing-selection-bias-input"
      RoutingWeightsOutput => "routing-weights-output"
      ExpertInput => "expert-input"
      ExpertOutput => "expert-output"
      AuxiliaryInput => "auxiliary-input"
      AuxiliaryOutput => "auxiliary-output"
      LogitsOutput => "logits-output"
      Workspace => "workspace"
    }
    text.write_string(role)
    text.write_char('@')
    text.write_string(item.byte_count().to_string())
    text.write_char('/')
    text.write_string(item.alignment().to_string())
  }
  text.to_string()
}
