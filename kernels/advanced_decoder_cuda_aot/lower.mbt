///|
pub fn AdvancedDecoderCudaAotLimits::new(
  maximum_tokens~ : Int,
  maximum_source_bytes~ : Int,
  maximum_recipe_bytes~ : Int,
) -> Self raise AdvancedDecoderCudaAotError {
  if maximum_tokens <= 0 ||
    maximum_tokens > 1048576 ||
    maximum_source_bytes <= 0 ||
    maximum_recipe_bytes <= 0 {
    raise Invalid(Limit)
  }
  { maximum_tokens, maximum_source_bytes, maximum_recipe_bytes, }
}

///|
fn checked_mul(
  left : Int64,
  right : Int64,
) -> Int64 raise AdvancedDecoderCudaAotError {
  if left <= 0L || right <= 0L || left > 9223372036854775807L / right {
    raise SizeOverflow
  }
  left * right
}

///|
fn operand(
  role : @advanced_decoder_artifact_admission.AdvancedDecoderLaunchOperandRole,
  byte_count : Int64,
) -> @advanced_decoder_artifact_admission.AdvancedDecoderLaunchOperand raise AdvancedDecoderCudaAotError {
  operand_with_alignment(role, byte_count, 16L)
}

///|
fn operand_with_alignment(
  role : @advanced_decoder_artifact_admission.AdvancedDecoderLaunchOperandRole,
  byte_count : Int64,
  alignment : Int64,
) -> @advanced_decoder_artifact_admission.AdvancedDecoderLaunchOperand raise AdvancedDecoderCudaAotError {
  @advanced_decoder_artifact_admission.AdvancedDecoderLaunchOperand::new(
    role~,
    byte_count~,
    alignment~,
  ) catch {
    error => raise Operand(error)
  }
}

///|
fn recipe(
  family : AdvancedDecoderFoundationAotFamily,
  numerical_contract : AdvancedDecoderFoundationNumericalContract,
  requirements : @advanced_decoder_kernel_requirements.AdvancedDecoderKernelRequirements,
  ordinal : Int,
  maximum_tokens : Int,
  target : @catalog.DeviceTarget,
  compiler : @luna_specialization.LunaCompilerPolicy,
  dimensions : @launch_contract.AotLaunchDimensions,
  operands : ArrayView[
    @advanced_decoder_artifact_admission.AdvancedDecoderLaunchOperand,
  ],
  source_digest : AdvancedDecoderCudaSourceDigest,
) -> Bytes {
  let compiler_identity = compiler.identity()
  let family_tag = match family {
    TokenEmbeddingBf16 => "token-embedding-bf16"
    QueryKeyRmsNormBf16 => "query-key-rms-norm-bf16"
    FinalRmsNormBf16 => "final-rms-norm-bf16"
  }
  let numerical_tag = match numerical_contract {
    BitExactBf16Copy => "bit-exact-bf16-copy"
    OrderedFloat32ReductionBf16RoundV1 => "ordered-f32-reduction-bf16-round-v1"
  }
  let operand_text = StringBuilder()
  for index, item in operands {
    if index > 0 {
      operand_text.write_char(',')
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
    operand_text.write_string(role)
    operand_text.write_char('@')
    operand_text.write_string(item.byte_count().to_string())
    operand_text.write_char('/')
    operand_text.write_string(item.alignment().to_string())
  }
  let geometry = requirements.geometry()
  @utf8.encode(
    "lunaflux.advanced-decoder-cuda-aot.v1" +
    "|family=" +
    family_tag +
    "|numerical=" +
    numerical_tag +
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
    "|geometry=" +
    geometry.vocabulary_size().to_string() +
    ":" +
    geometry.hidden_size().to_string() +
    ":" +
    geometry.attention_head_count().to_string() +
    ":" +
    geometry.kv_head_count().to_string() +
    ":" +
    geometry.attention_head_dimension().to_string() +
    ":" +
    geometry.rope_head_dimension().to_string() +
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
    operand_text.to_string() +
    "|source=" +
    source_digest.as_hex(),
  )
}

///|
/// Lower one exact shared foundation operation into deterministic CUDA source.
/// Only token embedding and final RMSNorm are admitted in this first slice.
pub fn lower_foundation_cuda_aot_candidate(
  requirements : @advanced_decoder_kernel_requirements.AdvancedDecoderKernelRequirements,
  requirement_ordinal~ : Int,
  maximum_tokens~ : Int,
  target~ : @catalog.DeviceTarget,
  compiler~ : @luna_specialization.LunaCompilerPolicy,
  limits~ : AdvancedDecoderCudaAotLimits,
) -> AdvancedDecoderFoundationCudaCandidate raise AdvancedDecoderCudaAotError {
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
  let geometry = requirements.geometry()
  let width = geometry.hidden_size()
  let activation_bytes = checked_mul(
    checked_mul(maximum_tokens.to_int64(), width.to_int64()),
    2L,
  )
  let (family, numerical_contract, symbol, source, dimensions, operands) = match
    (requirement.operation(), requirement.capability()) {
    (TokenEmbedding, EmbeddingLookupAot) => {
      let work_items = checked_mul(maximum_tokens.to_int64(), width.to_int64())
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
      let table_bytes = checked_mul(
        checked_mul(geometry.vocabulary_size().to_int64(), width.to_int64()),
        2L,
      )
      let operands = [
        operand(StepCountsInput, 20L),
        operand(TokenIdsInput, checked_mul(maximum_tokens.to_int64(), 4L)),
        operand(ModelWeights, table_bytes),
        operand(HiddenOutput, activation_bytes),
      ]
      let symbol = "lunaflux_advanced_decoder_embedding_bf16_v1"
      let rendered = @decoder_foundation_cuda_source.token_embedding_bf16_source(
        vocabulary_size=geometry.vocabulary_size(),
        hidden_size=width,
        function_symbol=symbol,
      ) catch {
        InvalidVocabularySize => raise Source(InvalidVocabularySize)
        InvalidRowCount => raise Source(InvalidRowCount)
        InvalidHiddenSize => raise Source(InvalidHiddenSize)
        InvalidEpsilon => raise Source(InvalidEpsilon)
        InvalidFunctionSymbol => raise Source(InvalidFunctionSymbol)
      }
      (
        TokenEmbeddingBf16,
        BitExactBf16Copy,
        symbol,
        rendered.source_bytes(),
        dimensions,
        operands,
      )
    }
    (QueryKeyRmsNorm(epsilon, query_width, key_value_width), QueryKeyRmsNormAot) => {
      if epsilon.is_nan() ||
        epsilon.is_inf() ||
        epsilon <= 0.0 ||
        epsilon > 1.0 ||
        query_width <= 0 ||
        key_value_width <= 0 {
        raise Invalid(Shape)
      }
      let dimensions = @launch_contract.AotLaunchDimensions::new(
        grid_x=maximum_tokens,
        grid_y=2,
        grid_z=1,
        block_x=256,
        block_y=1,
        block_z=1,
        shared_memory_bytes=1024,
      ) catch {
        error => raise Launch(error)
      }
      let query_bytes = checked_mul(
        checked_mul(maximum_tokens.to_int64(), query_width.to_int64()),
        2L,
      )
      let key_value_bytes = checked_mul(
        checked_mul(maximum_tokens.to_int64(), key_value_width.to_int64()),
        2L,
      )
      let operands = [
        operand_with_alignment(StepCountsInput, 20L, 4L),
        operand(QueryInput, query_bytes),
        operand(QueryNormWeights, checked_mul(query_width.to_int64(), 2L)),
        operand(QueryOutput, query_bytes),
        operand(KeyValueInput, key_value_bytes),
        operand(
          KeyValueNormWeights,
          checked_mul(key_value_width.to_int64(), 2L),
        ),
        operand(KeyValueOutput, key_value_bytes),
      ]
      let symbol = "lunaflux_advanced_decoder_query_key_rms_norm_bf16_v1"
      let rendered = @decoder_foundation_cuda_source.paired_rms_norm_bf16_source(
        first_width=query_width,
        second_width=key_value_width,
        maximum_rows=maximum_tokens,
        epsilon~,
        function_symbol=symbol,
      ) catch {
        InvalidVocabularySize => raise Source(InvalidVocabularySize)
        InvalidRowCount => raise Source(InvalidRowCount)
        InvalidHiddenSize => raise Source(InvalidHiddenSize)
        InvalidEpsilon => raise Source(InvalidEpsilon)
        InvalidFunctionSymbol => raise Source(InvalidFunctionSymbol)
      }
      (
        QueryKeyRmsNormBf16,
        OrderedFloat32ReductionBf16RoundV1,
        symbol,
        rendered.source_bytes(),
        dimensions,
        operands,
      )
    }
    (FinalRmsNorm(epsilon), RmsNormAot) => {
      let dimensions = @launch_contract.AotLaunchDimensions::new(
        grid_x=maximum_tokens,
        grid_y=1,
        grid_z=1,
        block_x=256,
        block_y=1,
        block_z=1,
        shared_memory_bytes=1024,
      ) catch {
        error => raise Launch(error)
      }
      let operands = [
        operand(StepCountsInput, 20L),
        operand(HiddenInput, activation_bytes),
        operand(ModelWeights, checked_mul(width.to_int64(), 2L)),
        operand(HiddenOutput, activation_bytes),
      ]
      let symbol = "lunaflux_advanced_decoder_final_rms_norm_bf16_v1"
      let rendered = @decoder_foundation_cuda_source.rms_norm_bf16_source(
        hidden_size=width,
        epsilon~,
        function_symbol=symbol,
      ) catch {
        InvalidVocabularySize => raise Source(InvalidVocabularySize)
        InvalidRowCount => raise Source(InvalidRowCount)
        InvalidHiddenSize => raise Source(InvalidHiddenSize)
        InvalidEpsilon => raise Source(InvalidEpsilon)
        InvalidFunctionSymbol => raise Source(InvalidFunctionSymbol)
      }
      (
        FinalRmsNormBf16,
        OrderedFloat32ReductionBf16RoundV1,
        symbol,
        rendered.source_bytes(),
        dimensions,
        operands,
      )
    }
    _ => raise UnsupportedRequirement(requirement_ordinal)
  }
  if source.length() > limits.maximum_source_bytes {
    raise Invalid(Encoding)
  }
  let source_digest = AdvancedDecoderCudaSourceDigest(
    @crypto.bytes_to_hex_string(@crypto.sha256(source)),
  )
  let encoded_recipe = recipe(
    family, numerical_contract, requirements, requirement_ordinal, maximum_tokens,
    target, compiler, dimensions, operands, source_digest,
  )
  if encoded_recipe.length() > limits.maximum_recipe_bytes {
    raise Invalid(Encoding)
  }
  {
    family,
    numerical_contract,
    source_identity: requirements.source_identity(),
    requirements_digest: requirements.digest(),
    requirement_ordinal,
    operation: requirement.operation(),
    capability: requirement.capability(),
    target,
    maximum_tokens,
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
