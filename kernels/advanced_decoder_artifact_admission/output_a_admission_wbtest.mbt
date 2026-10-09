///|
fn output_a_admission_requirements(
  hidden_size? : Int = 16,
  query_rank? : Int = 4,
  output_rank? : Int = 4,
  attention_head_dimension? : Int = 64,
) -> @advanced_decoder_kernel_requirements.AdvancedDecoderKernelRequirements {
  let shape = try! @advanced_decoder_execution_plan.AdvancedDecoderShape::new(
    vocabulary_size=128,
    hidden_size~,
    decoder_layer_count=1,
    attention_head_count=2,
    kv_head_count=1,
    attention_head_dimension~,
    rope_head_dimension=64,
    query_low_rank=query_rank,
    output_low_rank=output_rank,
    output_group_count=1,
    maximum_position_count=1024,
  )
  let low_rank = try! @advanced_decoder_execution_plan.LowRankAttentionPlan::new(
    query_rank~,
    output_rank~,
    output_group_count=1,
    query_key_norm_epsilon=0.001,
  )
  let position = try! @advanced_decoder_execution_plan.PositionEncodingPlan::new(
    rope_theta=10000.0,
    compressed_rope_theta=20000.0,
    original_position_count=128,
    scaling_factor=8.0,
    beta_fast=32.0,
    beta_slow=1.0,
  )
  let hyper = try! @advanced_decoder_execution_plan.HyperConnectionPlan::new(
    width=2,
    control_epsilon=0.000001,
    normalization_epsilon=0.000001,
    sinkhorn_iterations=4,
  )
  let moe = try! @advanced_decoder_execution_plan.MoeExecutionPlan::new(
    routed_expert_count=8,
    shared_expert_count=1,
    experts_per_token=2,
    intermediate_size=32,
    routed_scaling_factor=1.5,
    normalize_selected_probabilities=true,
    score_function=Sigmoid,
    selection_method=NoAuxiliaryLossTokenChoice,
    activation=SwiGlu,
  )
  let plan = try! @advanced_decoder_execution_plan.AdvancedDecoderExecutionPlan::new(
    schema_version=1,
    profile_identity="test/output-a-admission/v1",
    content_digest=try! @spec.ContentDigest::from_sha256(String::make(64, 'a')),
    shape~,
    low_rank_attention=low_rank,
    position_encoding=position,
    hyper_connection=hyper,
    moe~,
    layers=[
      try! @advanced_decoder_execution_plan.AdvancedDecoderLayerPlan::new(
        layer_index=0,
        attention=try! @advanced_decoder_execution_plan.AdvancedAttentionPlan::sliding_window_causal(
          64,
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
      maximum_layers=4,
      maximum_auxiliary_stages=4,
      maximum_requirements=64,
    ),
  )
}

///|
test "output-A artifact ABI is directional exact and output-B stays unsupported" {
  let requirements = output_a_admission_requirements()
  let mut ordinal = -1
  for index, requirement in requirements.requirements() {
    if requirement.operation() is AttentionOutputAProjectionV1(_, _, _, _) {
      ordinal = index
    }
  }
  assert_true(ordinal >= 0)
  let operation = requirements.requirements()[ordinal].operation()
  let dimensions = try! @launch_contract.AotLaunchDimensions::new(
    grid_x=2,
    grid_y=1,
    grid_z=1,
    block_x=256,
    block_y=1,
    block_z=1,
    shared_memory_bytes=0,
  )
  let exact = [
    try! AdvancedDecoderLaunchOperand::new(
      role=StepCountsInput,
      byte_count=20L,
      alignment=4L,
    ),
    try! AdvancedDecoderLaunchOperand::new(
      role=AttentionOutputProjectionInput,
      byte_count=512L,
      alignment=16L,
    ),
    try! AdvancedDecoderLaunchOperand::new(
      role=AttentionOutputAWeightsInput,
      byte_count=1024L,
      alignment=16L,
    ),
    try! AdvancedDecoderLaunchOperand::new(
      role=AttentionOutputRankOutput,
      byte_count=16L,
      alignment=16L,
    ),
  ]
  assert_true(
    operands_are_exact(
      operation,
      ReadOnlyArray::from_array(exact),
      dimensions,
      requirements.geometry(),
      8,
    ),
  )
  let wrong_role = exact.copy()
  wrong_role[2] = try! AdvancedDecoderLaunchOperand::new(
    role=ModelWeights,
    byte_count=1024L,
    alignment=16L,
  )
  assert_false(
    operands_are_exact(
      operation,
      ReadOnlyArray::from_array(wrong_role),
      dimensions,
      requirements.geometry(),
      8,
    ),
  )
  let wrong_bytes = exact.copy()
  wrong_bytes[3] = try! AdvancedDecoderLaunchOperand::new(
    role=AttentionOutputRankOutput,
    byte_count=14L,
    alignment=16L,
  )
  assert_false(
    operands_are_exact(
      operation,
      ReadOnlyArray::from_array(wrong_bytes),
      dimensions,
      requirements.geometry(),
      8,
    ),
  )
  assert_false(
    operands_are_exact(
      requirements.requirements()[ordinal + 1].operation(),
      [
        try! AdvancedDecoderLaunchOperand::new(
          role=HiddenInput,
          byte_count=1L,
          alignment=1L,
        ),
      ],
      dimensions,
      requirements.geometry(),
      8,
    ),
  )
}
