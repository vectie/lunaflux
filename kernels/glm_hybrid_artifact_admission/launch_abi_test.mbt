///|
fn artifact_test_exact_launch(
  grid_x : Int,
  shared_memory_bytes : Int,
) -> @launch_contract.AotLaunchDimensions {
  try! @launch_contract.AotLaunchDimensions::new(
    grid_x~,
    grid_y=1,
    grid_z=1,
    block_x=256,
    block_y=1,
    block_z=1,
    shared_memory_bytes~,
  )
}

///|
fn artifact_test_serial_launch() -> @launch_contract.AotLaunchDimensions {
  try! @launch_contract.AotLaunchDimensions::new(
    grid_x=1,
    grid_y=1,
    grid_z=1,
    block_x=1,
    block_y=1,
    block_z=1,
    shared_memory_bytes=0,
  )
}

///|
fn artifact_test_operand(
  role : GlmHybridLaunchOperandRole,
  bytes : Int64,
  alignment : Int64,
) -> GlmHybridLaunchOperand {
  try! GlmHybridLaunchOperand::new(role~, byte_count=bytes, alignment~)
}

///|
fn artifact_test_abi(
  operation : @glm_hybrid_kernel_requirements.GlmHybridSemanticOperation,
  ordinal : Int,
) -> (
  String,
  @launch_contract.AotLaunchDimensions,
  Array[GlmHybridLaunchOperand],
) {
  match operation {
    TokenEmbedding =>
      (
        "lunaflux_glm_hybrid_embedding_bf16_v1",
        artifact_test_exact_launch(1, 0),
        [
          artifact_test_operand(StepCountsInput, 20L, 4L),
          artifact_test_operand(TokenIdsInput, 8L, 4L),
          artifact_test_operand(ModelWeights, 1024L, 16L),
          artifact_test_operand(HiddenOutput, 32L, 16L),
        ],
      )
    DsaQueryAProjection(_, _) =>
      (
        "lunaflux_glm_hybrid_dsa_query_a_projection_bf16_v1",
        artifact_test_exact_launch(1, 0),
        [
          artifact_test_operand(StepCountsInput, 20L, 4L),
          artifact_test_operand(HiddenInput, 32L, 16L),
          artifact_test_operand(ModelWeights, 64L, 16L),
          artifact_test_operand(DsaQueryCompressedOutput, 16L, 16L),
        ],
      )
    DsaQueryARmsNorm(_, _) =>
      (
        "lunaflux_glm_hybrid_dsa_query_a_rms_norm_bf16_v1",
        artifact_test_exact_launch(2, 1024),
        [
          artifact_test_operand(StepCountsInput, 20L, 4L),
          artifact_test_operand(DsaQueryCompressedInput, 16L, 16L),
          artifact_test_operand(ModelWeights, 8L, 16L),
          artifact_test_operand(DsaQueryNormalizedOutput, 16L, 16L),
        ],
      )
    DsaQueryBProjection(_, _, _) =>
      (
        "lunaflux_glm_hybrid_dsa_query_b_projection_bf16_v1",
        artifact_test_exact_launch(1, 0),
        [
          artifact_test_operand(StepCountsInput, 20L, 4L),
          artifact_test_operand(DsaQueryNormalizedInput, 16L, 16L),
          artifact_test_operand(ModelWeights, 96L, 16L),
          artifact_test_operand(DsaQueryHeadsOutput, 48L, 16L),
        ],
      )
    DsaKeyValueAProjection(_, _, _) =>
      (
        "lunaflux_glm_hybrid_dsa_key_value_a_projection_bf16_v1",
        artifact_test_exact_launch(1, 0),
        [
          artifact_test_operand(StepCountsInput, 20L, 4L),
          artifact_test_operand(HiddenInput, 32L, 16L),
          artifact_test_operand(ModelWeights, 64L, 16L),
          artifact_test_operand(DsaKeyValueCompressedRopeOutput, 16L, 16L),
        ],
      )
    KeyValueLowRankProjection(_, _, _, _) =>
      (
        "lunaflux_glm_hybrid_dsa_key_value_b_projection_bf16_v1",
        artifact_test_exact_launch(1, 0),
        [
          artifact_test_operand(StepCountsInput, 20L, 4L),
          artifact_test_operand(DsaKeyValueNormalizedInput, 8L, 16L),
          artifact_test_operand(ModelWeights, 32L, 16L),
          artifact_test_operand(DsaKeyValueHeadsOutput, 32L, 16L),
        ],
      )
    QueryKeyRmsNorm(_) =>
      (
        "lunaflux_glm_hybrid_dsa_kv_a_split_rms_norm_bf16_v1",
        artifact_test_exact_launch(2, 1024),
        [
          artifact_test_operand(StepCountsInput, 20L, 4L),
          artifact_test_operand(DsaKeyValueCompressedRopeInput, 16L, 16L),
          artifact_test_operand(ModelWeights, 4L, 16L),
          artifact_test_operand(DsaKeyValueNormalizedOutput, 8L, 16L),
          artifact_test_operand(DsaKeyRopeSuffixOutput, 8L, 16L),
        ],
      )
    GlmRotaryPosition(_, _) =>
      (
        "lunaflux_glm_hybrid_dsa_interleaved_rope_bf16_v1",
        try! @launch_contract.AotLaunchDimensions::new(
          grid_x=2,
          grid_y=3,
          grid_z=1,
          block_x=256,
          block_y=1,
          block_z=1,
          shared_memory_bytes=0,
        ),
        [
          artifact_test_operand(StepCountsInput, 20L, 4L),
          artifact_test_operand(QueryPositionsInput, 8L, 4L),
          artifact_test_operand(DsaQueryHeadsOutput, 48L, 16L),
          artifact_test_operand(DsaKeyRopeSuffixOutput, 8L, 16L),
          artifact_test_operand(DsaAttentionQueriesInput, 48L, 16L),
          artifact_test_operand(DsaAttentionKeyRopeInput, 8L, 16L),
        ],
      )
    DsaKeyValueAttentionAssembly(_, _, _, _) =>
      (
        "lunaflux_glm_hybrid_dsa_kv_assembly_rope_bf16_v1",
        artifact_test_exact_launch(1, 0),
        [
          artifact_test_operand(StepCountsInput, 20L, 4L),
          artifact_test_operand(DsaKeyValueHeadsOutput, 32L, 16L),
          artifact_test_operand(DsaAttentionKeyRopeInput, 8L, 16L),
          artifact_test_operand(DsaAttentionKeysInput, 24L, 16L),
          artifact_test_operand(DsaAttentionValuesInput, 16L, 16L),
        ],
      )
    DenseSwiGlu(_, _) =>
      (
        "lunaflux_glm_hybrid_dense_swiglu_bf16_v1",
        artifact_test_exact_launch(1, 0),
        [
          artifact_test_operand(StepCountsInput, 20L, 4L),
          artifact_test_operand(HiddenInput, 32L, 16L),
          artifact_test_operand(DenseGateWeights, 256L, 16L),
          artifact_test_operand(DenseUpWeights, 256L, 16L),
          artifact_test_operand(DenseDownWeights, 256L, 16L),
          artifact_test_operand(HiddenOutput, 32L, 16L),
        ],
      )
    DecoderRmsNorm(_, _) =>
      (
        "lunaflux_glm_hybrid_decoder_rms_norm_bf16_v1",
        artifact_test_exact_launch(2, 1024),
        [
          artifact_test_operand(StepCountsInput, 20L, 4L),
          artifact_test_operand(HiddenInput, 32L, 16L),
          artifact_test_operand(ModelWeights, 16L, 16L),
          artifact_test_operand(HiddenOutput, 32L, 16L),
        ],
      )
    FinalRmsNorm(_, _) =>
      (
        "lunaflux_glm_hybrid_final_rms_norm_bf16_v1",
        artifact_test_exact_launch(2, 1024),
        [
          artifact_test_operand(StepCountsInput, 20L, 4L),
          artifact_test_operand(HiddenInput, 32L, 16L),
          artifact_test_operand(ModelWeights, 16L, 16L),
          artifact_test_operand(HiddenOutput, 32L, 16L),
        ],
      )
    AttentionOutputProjection(_, _) =>
      (
        "lunaflux_glm_hybrid_attention_output_projection_bf16_v1",
        artifact_test_exact_launch(1, 0),
        [
          artifact_test_operand(StepCountsInput, 20L, 4L),
          artifact_test_operand(HiddenInput, 32L, 16L),
          artifact_test_operand(ModelWeights, 128L, 16L),
          artifact_test_operand(HiddenOutput, 32L, 16L),
        ],
      )
    DsaIndexPooling(_, _, _) =>
      (
        "lunaflux_glm_hybrid_dsa_key_pool_bf16_v1",
        artifact_test_serial_launch(),
        [
          artifact_test_operand(StepCountsInput, 20L, 4L),
          artifact_test_operand(DsaIndexKeysInput, 64L, 16L),
          artifact_test_operand(DsaIndexGateScoresInput, 64L, 16L),
          artifact_test_operand(DsaIndexValidityInput, 32L, 4L),
          artifact_test_operand(ModelWeights, 16L, 16L),
          artifact_test_operand(DsaPooledKeysOutput, 32L, 16L),
          artifact_test_operand(DsaPooledRawIndicesOutput, 32L, 4L),
          artifact_test_operand(DsaPooledValidityOutput, 16L, 4L),
          artifact_test_operand(DsaPooledCountOutput, 4L, 4L),
        ],
      )
    DsaIndexCompute(_, _, _, _) =>
      (
        "lunaflux_glm_hybrid_dsa_index_select_bf16_v1",
        artifact_test_serial_launch(),
        [
          artifact_test_operand(StepCountsInput, 20L, 4L),
          artifact_test_operand(QueryPositionsInput, 32L, 4L),
          artifact_test_operand(DsaIndexQueriesInput, 128L, 16L),
          artifact_test_operand(DsaIndexHeadWeightsInput, 32L, 16L),
          artifact_test_operand(DsaCandidateKeysInput, 32L, 16L),
          artifact_test_operand(DsaCandidateRawIndicesInput, 32L, 4L),
          artifact_test_operand(DsaCandidateValidityInput, 16L, 4L),
          artifact_test_operand(DsaCandidateCountInput, 4L, 4L),
          artifact_test_operand(DsaKeyValidityInput, 32L, 4L),
          artifact_test_operand(DsaSelectedIndicesOutput, 288L, 4L),
        ],
      )
    DsaIndexReuse =>
      (
        "lunaflux_glm_hybrid_dsa_index_reuse_i32_v1",
        artifact_test_exact_launch(1, 0),
        [
          artifact_test_operand(StepCountsInput, 20L, 4L),
          artifact_test_operand(DsaSelectedIndicesInput, 288L, 4L),
          artifact_test_operand(DsaSelectedIndicesOutput, 288L, 4L),
        ],
      )
    DsaSparseAttention(_, _, _, _) =>
      (
        "lunaflux_glm_hybrid_dsa_sparse_attention_bf16_v1",
        artifact_test_serial_launch(),
        [
          artifact_test_operand(StepCountsInput, 20L, 4L),
          artifact_test_operand(QueryPositionsInput, 32L, 4L),
          artifact_test_operand(DsaKeyCountInput, 4L, 4L),
          artifact_test_operand(DsaKeyValidityInput, 32L, 4L),
          artifact_test_operand(DsaAttentionQueriesInput, 192L, 16L),
          artifact_test_operand(DsaAttentionKeysInput, 96L, 16L),
          artifact_test_operand(DsaAttentionValuesInput, 64L, 16L),
          artifact_test_operand(DsaSelectedIndicesInput, 288L, 4L),
          artifact_test_operand(DsaAttentionOutput, 128L, 16L),
        ],
      )
    LanguageModelHead(_, _) =>
      (
        "lunaflux_glm_hybrid_language_model_head_bf16_v1",
        artifact_test_exact_launch(1, 0),
        [
          artifact_test_operand(StepCountsInput, 20L, 4L),
          artifact_test_operand(HiddenInput, 32L, 16L),
          artifact_test_operand(ModelWeights, 1024L, 16L),
          artifact_test_operand(LogitsOutput, 256L, 16L),
        ],
      )
    MoeRouterProjection(_, _, _) => artifact_test_moe_router_abi()
    MoeSigmoidAndCorrectionBias(_) => artifact_test_moe_score_abi()
    MoeGroupedNoAuxiliaryLossTopK(_, _, _, _, _, _, _) =>
      artifact_test_moe_grouped_routing_abi()
    RoutedExpertFeedForward(_, _, _) => artifact_test_moe_routed_expert_abi()
    SharedExpertFeedForward(_, _, _) => artifact_test_moe_shared_expert_abi()
    RoutedSharedExpertCombine(_, _, _) => artifact_test_moe_combine_abi()
    MoeDecoderResidualAdd(_) =>
      (
        "lunaflux_glm_hybrid_moe_decoder_residual_add_bf16_v1",
        artifact_test_exact_launch(1, 0),
        [
          artifact_test_operand(StepCountsInput, 20L, 4L),
          artifact_test_operand(DecoderResidualInput, 32L, 16L),
          artifact_test_operand(MoeCombinedOutput, 32L, 16L),
          artifact_test_operand(DecoderResidualOutput, 32L, 16L),
        ],
      )
    _ =>
      (
        "lunaflux_glm_hybrid_test_op_" + ordinal.to_string(),
        artifact_test_launch(),
        [artifact_test_operand(HiddenInput, 16L, 16L)],
      )
  }
}

///|
test "function symbol and exact executable operand ABI fail closed" {
  let requirements = artifact_test_requirements()
  let target = artifact_test_target()
  let wrong_symbol = artifact_test_claims(
    requirements,
    target,
    qualified=true,
    wrong_function_symbol=true,
  )
  try admit(requirements, target, wrong_symbol, artifact_test_limits()) catch {
    InvalidFunctionSymbol(ordinal=0) => ()
    _ => fail("unexpected function-symbol error")
  } noraise {
    _ => fail("invalid function symbol admitted")
  }
  let wrong_operands = artifact_test_claims(
    requirements,
    target,
    qualified=true,
    wrong_launch_operands=true,
  )
  try
    admit(requirements, target, wrong_operands, artifact_test_limits())
  catch {
    InvalidLaunchOperands(ordinal=0) => ()
    _ => fail("unexpected launch-operand error")
  } noraise {
    _ => fail("misaligned step counts admitted")
  }
  let wrong_projection = artifact_test_claims(
    requirements,
    target,
    qualified=true,
    wrong_dsa_projection_operands=true,
  )
  try
    admit(requirements, target, wrong_projection, artifact_test_limits())
  catch {
    InvalidLaunchOperands(ordinal=2) => ()
    _ => fail("unexpected DSA input-projection operand error")
  } noraise {
    _ => fail("generic hidden-output role admitted for q-a projection")
  }
  let wrong_query_norm = artifact_test_claims(
    requirements,
    target,
    qualified=true,
    wrong_query_a_norm_operands=true,
  )
  try
    admit(requirements, target, wrong_query_norm, artifact_test_limits())
  catch {
    InvalidLaunchOperands(ordinal=3) => ()
    _ => fail("unexpected q-a RMSNorm operand error")
  } noraise {
    _ => fail("generic hidden-output role admitted for q-a RMSNorm")
  }
  let wrong_query_b = artifact_test_claims(
    requirements,
    target,
    qualified=true,
    wrong_query_b_operands=true,
  )
  try admit(requirements, target, wrong_query_b, artifact_test_limits()) catch {
    InvalidLaunchOperands(ordinal=4) => ()
    _ => fail("unexpected q-b projection operand error")
  } noraise {
    _ => fail("generic hidden-output role admitted for q-b projection")
  }
  let wrong_kv_b = artifact_test_claims(
    requirements,
    target,
    qualified=true,
    wrong_key_value_b_operands=true,
  )
  try admit(requirements, target, wrong_kv_b, artifact_test_limits()) catch {
    InvalidLaunchOperands(ordinal=7) => ()
    _ => fail("unexpected kv-b projection operand error")
  } noraise {
    _ => fail("generic hidden-output role admitted for kv-b projection")
  }
  let wrong_kv_norm = artifact_test_claims(
    requirements,
    target,
    qualified=true,
    wrong_kv_split_norm_operands=true,
  )
  try admit(requirements, target, wrong_kv_norm, artifact_test_limits()) catch {
    InvalidLaunchOperands(ordinal=8) => ()
    _ => fail("unexpected KV split-norm operand error")
  } noraise {
    _ => fail("missing Full rope-suffix output admitted")
  }
  let wrong_rotary = artifact_test_claims(
    requirements,
    target,
    qualified=true,
    wrong_rotary_operands=true,
  )
  try admit(requirements, target, wrong_rotary, artifact_test_limits()) catch {
    InvalidLaunchOperands(ordinal=9) => ()
    _ => fail("unexpected rotary operand error")
  } noraise {
    _ => fail("unrotated key-suffix role admitted as rotary output")
  }
  let wrong_dense_swiglu = artifact_test_claims(
    requirements,
    target,
    qualified=true,
    wrong_dense_swiglu_operands=true,
  )
  let mut dense_ordinal = -1
  for ordinal, item in requirements.requirements() {
    if item.operation() is DenseSwiGlu(_, _) {
      dense_ordinal = ordinal
    }
  }
  try
    admit(requirements, target, wrong_dense_swiglu, artifact_test_limits())
  catch {
    InvalidLaunchOperands(ordinal~) => assert_eq(ordinal, dense_ordinal)
    _ => fail("unexpected dense SwiGLU operand error")
  } noraise {
    _ => fail("swapped dense gate/up weights admitted")
  }
}
