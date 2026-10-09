///|
fn deterministic_index_test_operand(
  role : AdvancedDecoderLaunchOperandRole,
  byte_count : Int64,
) -> AdvancedDecoderLaunchOperand {
  try! AdvancedDecoderLaunchOperand::new(role~, byte_count~, alignment=4L)
}

///|
fn deterministic_index_test_operands() -> Array[AdvancedDecoderLaunchOperand] {
  [
    deterministic_index_test_operand(StepCountsInput, 20L),
    deterministic_index_test_operand(CompressedQueryPositionsInput, 8L),
    deterministic_index_test_operand(CompressedCacheOffsetsInput, 8L),
    deterministic_index_test_operand(CompressedIndexCountsOutput, 8L),
    deterministic_index_test_operand(CompressedIndicesOutput, 64L),
  ]
}

///|
fn deterministic_index_test_dimensions(
  rows : Int,
) -> @launch_contract.AotLaunchDimensions {
  try! @launch_contract.AotLaunchDimensions::new(
    grid_x=rows,
    grid_y=1,
    grid_z=1,
    block_x=256,
    block_y=1,
    block_z=1,
    shared_memory_bytes=0,
  )
}

///|
fn deterministic_index_test_requirements() -> @advanced_decoder_kernel_requirements.AdvancedDecoderKernelRequirements {
  let shape = try! @advanced_decoder_execution_plan.AdvancedDecoderShape::new(
    vocabulary_size=128,
    hidden_size=16,
    decoder_layer_count=1,
    attention_head_count=2,
    kv_head_count=1,
    attention_head_dimension=64,
    rope_head_dimension=64,
    query_low_rank=4,
    output_low_rank=4,
    output_group_count=1,
    maximum_position_count=1024,
  )
  let plan = try! @advanced_decoder_execution_plan.AdvancedDecoderExecutionPlan::new(
    schema_version=1,
    profile_identity="test/deterministic-index-artifact-abi/v1",
    content_digest=try! @spec.ContentDigest::from_sha256(String::make(64, 'd')),
    shape~,
    low_rank_attention=try! @advanced_decoder_execution_plan.LowRankAttentionPlan::new(
      query_rank=4,
      output_rank=4,
      output_group_count=1,
      query_key_norm_epsilon=0.000001,
    ),
    position_encoding=try! @advanced_decoder_execution_plan.PositionEncodingPlan::new(
      rope_theta=10000.0,
      compressed_rope_theta=20000.0,
      original_position_count=128,
      scaling_factor=8.0,
      beta_fast=32.0,
      beta_slow=1.0,
    ),
    hyper_connection=try! @advanced_decoder_execution_plan.HyperConnectionPlan::new(
      width=2,
      control_epsilon=0.000001,
      normalization_epsilon=0.000001,
      sinkhorn_iterations=4,
    ),
    moe=try! @advanced_decoder_execution_plan.MoeExecutionPlan::new(
      routed_expert_count=8,
      shared_expert_count=1,
      experts_per_token=2,
      intermediate_size=32,
      routed_scaling_factor=1.5,
      normalize_selected_probabilities=true,
      score_function=Sigmoid,
      selection_method=NoAuxiliaryLossTokenChoice,
      activation=SwiGlu,
    ),
    layers=[
      try! @advanced_decoder_execution_plan.AdvancedDecoderLayerPlan::new(
        layer_index=0,
        attention=try! @advanced_decoder_execution_plan.AdvancedAttentionPlan::deterministic_compressed(
          compression_ratio=128,
          window_size=64,
        ),
      ),
    ],
    auxiliary_prediction=try! @advanced_decoder_execution_plan.AuxiliaryPredictionPlan::multi_token_prediction(
      1,
    ),
  )
  try! @advanced_decoder_kernel_requirements.derive_requirements(
    plan,
    limits=try! @advanced_decoder_kernel_requirements.AdvancedDecoderKernelRequirementLimits::new(
      maximum_layers=2,
      maximum_auxiliary_stages=2,
      maximum_requirements=64,
    ),
  )
}

///|
test "deterministic index artifact ABI binds directional roles and exact bytes" {
  let requirements = deterministic_index_test_requirements()
  let dimensions = deterministic_index_test_dimensions(2)
  let operands = deterministic_index_test_operands()
  assert_eq(expected_operand_roles(DeterministicCompressedIndex(128)), [
    StepCountsInput,
    CompressedQueryPositionsInput,
    CompressedCacheOffsetsInput,
    CompressedIndexCountsOutput,
    CompressedIndicesOutput,
  ])
  assert_true(
    operands_are_exact(
      DeterministicCompressedIndex(128),
      ReadOnlyArray::from_array(operands),
      dimensions,
      requirements.geometry(),
      8,
    ),
  )
}

///|
test "deterministic index artifact ABI rejects wrong role bytes and launch" {
  let requirements = deterministic_index_test_requirements()
  let dimensions = deterministic_index_test_dimensions(2)
  let wrong_role = deterministic_index_test_operands()
  wrong_role[2] = deterministic_index_test_operand(PositionInput, 8L)
  assert_false(
    operands_are_exact(
      DeterministicCompressedIndex(128),
      ReadOnlyArray::from_array(wrong_role),
      dimensions,
      requirements.geometry(),
      8,
    ),
  )
  let wrong_bytes = deterministic_index_test_operands()
  wrong_bytes[4] = deterministic_index_test_operand(
    CompressedIndicesOutput,
    60L,
  )
  assert_false(
    operands_are_exact(
      DeterministicCompressedIndex(128),
      ReadOnlyArray::from_array(wrong_bytes),
      dimensions,
      requirements.geometry(),
      8,
    ),
  )
  assert_false(
    operands_are_exact(
      DeterministicCompressedIndex(128),
      ReadOnlyArray::from_array(deterministic_index_test_operands()),
      deterministic_index_test_dimensions(3),
      requirements.geometry(),
      8,
    ),
  )
}
