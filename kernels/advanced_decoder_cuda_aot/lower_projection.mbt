///|
fn projection_operand_text(
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

///|
fn projection_recipe(
  requirements : @advanced_decoder_kernel_requirements.AdvancedDecoderKernelRequirements,
  ordinal : Int,
  maximum_tokens : Int,
  input_width : Int,
  output_width : Int,
  target : @catalog.DeviceTarget,
  compiler : @luna_specialization.LunaCompilerPolicy,
  dimensions : @launch_contract.AotLaunchDimensions,
  operands : ArrayView[
    @advanced_decoder_artifact_admission.AdvancedDecoderLaunchOperand,
  ],
  source_digest : AdvancedDecoderCudaSourceDigest,
) -> Bytes {
  let compiler_identity = compiler.identity()
  @utf8.encode(
    "lunaflux.advanced-decoder-projection-cuda-aot.v1" +
    "|family=language-model-head-bf16" +
    "|numerical=ordered-f32-accumulation-bf16-round-v1" +
    "|content=" +
    requirements.source_identity().content().as_hex() +
    "|plan=" +
    requirements.source_identity().plan().as_hex() +
    "|requirements=" +
    requirements.digest().as_hex() +
    "|ordinal=" +
    ordinal.to_string() +
    "|tokens=" +
    maximum_tokens.to_string() +
    "|projection=" +
    input_width.to_string() +
    ":" +
    output_width.to_string() +
    "|sm=" +
    target.compute_major().to_string() +
    "." +
    target.compute_minor().to_string() +
    ":" +
    target.supports_bf16().to_string() +
    ":" +
    target.supports_cublas_lt().to_string() +
    "|compiler=" +
    compiler_identity.toolchain_digest_sha256() +
    ":" +
    compiler_identity.major().to_string() +
    "." +
    compiler_identity.minor().to_string() +
    "." +
    compiler_identity.patch().to_string() +
    "|opt=" +
    compiler.optimization_level().to_string() +
    "|fma=" +
    compiler.fused_multiply_add().to_string() +
    "|reassociate=" +
    compiler.reassociate().to_string() +
    "|max-registers=" +
    compiler.max_registers().to_string() +
    "|launch=" +
    dimensions.grid_x().to_string() +
    ":" +
    dimensions.grid_y().to_string() +
    ":" +
    dimensions.grid_z().to_string() +
    ":" +
    dimensions.block_x().to_string() +
    ":" +
    dimensions.block_y().to_string() +
    ":" +
    dimensions.block_z().to_string() +
    ":" +
    dimensions.shared_memory_bytes().to_string() +
    "|operands=" +
    projection_operand_text(operands) +
    "|source=" +
    source_digest.as_hex(),
  )
}

///|
/// Lowers the terminal BF16 vocabulary projection into deterministic,
/// correctness-grade CUDA source. The candidate is intentionally non-bindable
/// until offline compilation and independent qualification authenticate it.
pub fn lower_language_model_head_cuda_aot_candidate(
  requirements : @advanced_decoder_kernel_requirements.AdvancedDecoderKernelRequirements,
  requirement_ordinal~ : Int,
  maximum_tokens~ : Int,
  target~ : @catalog.DeviceTarget,
  compiler~ : @luna_specialization.LunaCompilerPolicy,
  limits~ : AdvancedDecoderCudaAotLimits,
) -> AdvancedDecoderProjectionCudaCandidate raise AdvancedDecoderCudaAotError {
  if maximum_tokens <= 0 || maximum_tokens > limits.maximum_tokens {
    raise Invalid(Limit)
  }
  if !target.supports_bf16() {
    raise Invalid(Target)
  }
  if compiler.reassociate() {
    raise Invalid(Compiler)
  }
  guard requirements.requirements().get(requirement_ordinal)
    is Some(requirement) else {
    raise UnsupportedRequirement(requirement_ordinal)
  }
  guard (requirement.operation(), requirement.capability())
    is (LanguageModelHead, DenseProjectionAot) else {
    raise UnsupportedRequirement(requirement_ordinal)
  }
  let geometry = requirements.geometry()
  let input_width = geometry.hidden_size()
  let output_width = geometry.vocabulary_size()
  let input_bytes = checked_mul(
    checked_mul(maximum_tokens.to_int64(), input_width.to_int64()),
    2L,
  )
  let weight_bytes = checked_mul(
    checked_mul(output_width.to_int64(), input_width.to_int64()),
    2L,
  )
  let output_bytes = checked_mul(
    checked_mul(maximum_tokens.to_int64(), output_width.to_int64()),
    2L,
  )
  if input_bytes > 2147483647L ||
    weight_bytes > 2147483647L ||
    output_bytes > 2147483647L {
    raise SizeOverflow
  }
  let work_items = checked_mul(
    maximum_tokens.to_int64(),
    output_width.to_int64(),
  )
  if work_items > 9223372036854775552L {
    raise SizeOverflow
  }
  let grid = (work_items + 255L) / 256L
  if grid > 2147483647L {
    raise SizeOverflow
  }
  let dimensions = @launch_contract.AotLaunchDimensions::new(
    grid_x=grid.to_int(),
    grid_y=1,
    grid_z=1,
    block_x=256,
    block_y=1,
    block_z=1,
    shared_memory_bytes=0,
  ) catch {
    error => raise Launch(error)
  }
  let operands = [
    operand_with_alignment(StepCountsInput, 20L, 4L),
    operand(HiddenInput, input_bytes),
    operand(ModelWeights, weight_bytes),
    operand(LogitsOutput, output_bytes),
  ]
  let symbol = "lunaflux_advanced_decoder_language_model_head_bf16_v1"
  let rendered = @decoder_projection_cuda_source.dense_projection_bf16_source(
    input_width~,
    output_width~,
    maximum_rows=maximum_tokens,
    function_symbol=symbol,
  ) catch {
    error => raise ProjectionSource(error)
  }
  let source = rendered.source_bytes()
  if source.length() > limits.maximum_source_bytes {
    raise Invalid(Encoding)
  }
  let source_digest = AdvancedDecoderCudaSourceDigest(
    @crypto.bytes_to_hex_string(@crypto.sha256(source)),
  )
  let encoded_recipe = projection_recipe(
    requirements, requirement_ordinal, maximum_tokens, input_width, output_width,
    target, compiler, dimensions, operands, source_digest,
  )
  if encoded_recipe.length() > limits.maximum_recipe_bytes {
    raise Invalid(Encoding)
  }
  {
    family: LanguageModelHeadBf16,
    numerical_contract: OrderedF32AccumulationBf16RoundV1,
    source_identity: requirements.source_identity(),
    requirements_digest: requirements.digest(),
    requirement_ordinal,
    operation: requirement.operation(),
    capability: requirement.capability(),
    target,
    maximum_tokens,
    input_width,
    output_width,
    dimensions,
    operands: ReadOnlyArray::from_array(operands),
    compiler,
    function_symbol: symbol,
    source,
    source_digest,
    recipe: encoded_recipe,
    recipe_digest: AdvancedDecoderCudaRecipeDigest(
      @crypto.bytes_to_hex_string(@crypto.sha256(encoded_recipe)),
    ),
  }
}
