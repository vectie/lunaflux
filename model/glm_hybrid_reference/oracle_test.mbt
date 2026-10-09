///|
fn tiny_oracle_limits() -> @glm_hybrid_reference.GlmTinyOracleLimits {
  try! @glm_hybrid_reference.GlmTinyOracleLimits::new(
    maximum_tokens=8,
    maximum_width=8,
    maximum_low_rank=8,
    maximum_intermediate=16,
    maximum_experts=8,
    maximum_selected=4,
    maximum_values=4096,
  )
}

///|
fn approximately_tiny(actual : Double, expected : Double) -> Bool {
  (actual - expected).abs() < 0.0000001
}

///|
fn dense_block_weights(
  attention_norm : Double,
  attention_output : Double,
  ffn_norm : Double,
  gate : Double,
  up : Double,
  down : Double,
  hyper_mixing? : ArrayView[Double] = [],
) -> @glm_hybrid_reference.GlmTinyHybridBlockWeights {
  @glm_hybrid_reference.GlmTinyHybridBlockWeights::new(
    attention_norm=[attention_norm],
    query_a=[1.0],
    query_b=[1.0],
    key_a=[1.0],
    key_b=[1.0],
    value_a=[1.0],
    value_b=[1.0],
    attention_output=[attention_output],
    ffn_norm=[ffn_norm],
    feed_forward=@glm_hybrid_reference.GlmTinyFeedForwardWeights::dense(
      gate=[gate],
      up=[up],
      down=[down],
    ),
    hyper_mixing~,
  )
}

///|
test "tiny DSA block matches independent RMSNorm low-rank sparse attention calculation" {
  let plan = @glm_hybrid_reference.GlmTinyHybridBlockPlan::new(
    token_count=2,
    head_count=1,
    head_dimension=1,
    low_rank=1,
    sparse_top_k=1,
    intermediate_size=1,
    rms_epsilon=1.0,
    feed_forward=Dense,
    composition=Residual,
    limits=tiny_oracle_limits(),
  )
  let weights = dense_block_weights(1.0, 2.0, 1.0, 0.0, 1.0, 1.0)
  let first = @glm_hybrid_reference.execute_tiny_hybrid_block(
    [1.0, 2.0],
    [0, 0],
    plan,
    weights,
    limits=tiny_oracle_limits(),
  )
  let normalized_first = 1.0 / 2.0.sqrt()
  assert_true(
    approximately_tiny(first.attention_branch()[0], 2.0 * normalized_first),
  )
  assert_true(
    approximately_tiny(first.attention_branch()[1], 2.0 * normalized_first),
  )
  assert_true(
    approximately_tiny(first.output()[0], 1.0 + 2.0 * normalized_first),
  )
  assert_true(
    approximately_tiny(first.output()[1], 2.0 + 2.0 * normalized_first),
  )
  assert_eq(first.feed_forward_branch(), [0.0, 0.0])
  let second = @glm_hybrid_reference.execute_tiny_hybrid_block(
    [1.0, 2.0],
    [0, 0],
    plan,
    weights,
    limits=tiny_oracle_limits(),
  )
  assert_eq(first.output(), second.output())
}

///|
test "dense SwiGLU mHC and F32 corrected MoE routing match hand formulas" {
  let hyper_plan = @glm_hybrid_reference.GlmTinyHybridBlockPlan::new(
    token_count=1,
    head_count=1,
    head_dimension=1,
    low_rank=1,
    sparse_top_k=1,
    intermediate_size=1,
    rms_epsilon=1.0,
    feed_forward=Dense,
    composition=MultiStreamHyperConnection(
      stream_count=2,
      branch_stream=0,
      output_stream=1,
    ),
    limits=tiny_oracle_limits(),
  )
  let hyper_weights = dense_block_weights(
    2.0.sqrt(),
    1.0,
    5.0.sqrt(),
    1.0,
    1.0,
    1.0,
    hyper_mixing=[1.0, 0.0, 0.25, 0.75],
  )
  let hyper = @glm_hybrid_reference.execute_tiny_hybrid_block(
    [1.0, 3.0],
    [0],
    hyper_plan,
    hyper_weights,
    limits=tiny_oracle_limits(),
  )
  let swiglu = 4.0 / (1.0 + @math.exp(-2.0))
  let branch_output = 2.0 + swiglu
  assert_true(approximately_tiny(hyper.feed_forward_branch()[0], swiglu))
  assert_true(
    approximately_tiny(hyper.output()[0], 0.25 * branch_output + 2.25),
  )

  let moe_plan = @glm_hybrid_reference.GlmTinyHybridBlockPlan::new(
    token_count=1,
    head_count=1,
    head_dimension=1,
    low_rank=1,
    sparse_top_k=1,
    intermediate_size=1,
    rms_epsilon=1.0,
    feed_forward=Moe(
      expert_count=2,
      experts_per_token=1,
      score_function=Sigmoid,
      normalize_selected=true,
      route_scale=1.0F,
    ),
    composition=Residual,
    limits=tiny_oracle_limits(),
  )
  let moe_weights = @glm_hybrid_reference.GlmTinyHybridBlockWeights::new(
    attention_norm=[2.0.sqrt()],
    query_a=[1.0],
    query_b=[1.0],
    key_a=[1.0],
    key_b=[1.0],
    value_a=[1.0],
    value_b=[1.0],
    attention_output=[1.0],
    ffn_norm=[5.0.sqrt()],
    feed_forward=@glm_hybrid_reference.GlmTinyFeedForwardWeights::moe(
      router=[0.0F, 0.0F],
      correction=[0.0F, 1.0F],
      gate=[10.0, 1.0],
      up=[10.0, 1.0],
      down=[10.0, 1.0],
    ),
    hyper_mixing=[],
  )
  let moe = @glm_hybrid_reference.execute_tiny_hybrid_block(
    [1.0],
    [0],
    moe_plan,
    moe_weights,
    limits=tiny_oracle_limits(),
  )
  assert_eq(moe.selected_experts(), [1])
  assert_true(approximately_tiny(moe.feed_forward_branch()[0], swiglu))
  assert_true(approximately_tiny(moe.output()[0], branch_output))
}

///|
test "tiny Flash vision attention MLP merge and projector match independent fixture" {
  let plan = @glm_hybrid_reference.GlmTinyVisionBlockPlan::new(
    patch_count=4,
    head_count=1,
    head_dimension=1,
    intermediate_size=1,
    merge_size=2,
    project_width=1,
    rms_epsilon=1.0,
    limits=tiny_oracle_limits(),
  )
  let weights = @glm_hybrid_reference.GlmTinyVisionBlockWeights::new(
    attention_norm=[2.0.sqrt()],
    query=[1.0],
    key=[1.0],
    value=[1.0],
    attention_output=[1.0],
    ffn_norm=[5.0.sqrt()],
    gate=[1.0],
    up=[1.0],
    down=[1.0],
    merge=[0.25, 0.25, 0.25, 0.25],
    merge_bias=[0.0],
    project=[2.0],
    project_bias=[1.0],
  )
  let result = @glm_hybrid_reference.execute_tiny_vision_block(
    [1.0, 1.0, 1.0, 1.0],
    plan,
    weights,
    limits=tiny_oracle_limits(),
  )
  let block_value = 2.0 + 4.0 / (1.0 + @math.exp(-2.0))
  assert_eq(result.row_count(), 1)
  assert_eq(result.width(), 1)
  assert_true(approximately_tiny(result.values()[0], 2.0 * block_value + 1.0))
}

///|
test "tiny block rejects bounds future duplicate padded-empty NaN and shape attacks" {
  try
    @glm_hybrid_reference.GlmTinyHybridBlockPlan::new(
      token_count=9,
      head_count=1,
      head_dimension=1,
      low_rank=1,
      sparse_top_k=1,
      intermediate_size=1,
      rms_epsilon=1.0,
      feed_forward=Dense,
      composition=Residual,
      limits=tiny_oracle_limits(),
    )
  catch {
    BoundsExceeded => ()
    _ => fail("unexpected bounds failure")
  } noraise {
    _ => fail("oversized oracle plan admitted")
  }
  let plan = @glm_hybrid_reference.GlmTinyHybridBlockPlan::new(
    token_count=2,
    head_count=1,
    head_dimension=1,
    low_rank=1,
    sparse_top_k=2,
    intermediate_size=1,
    rms_epsilon=1.0,
    feed_forward=Dense,
    composition=Residual,
    limits=tiny_oracle_limits(),
  )
  let weights = dense_block_weights(1.0, 1.0, 1.0, 0.0, 1.0, 1.0)
  let padded = @glm_hybrid_reference.execute_tiny_hybrid_block(
    [1.0, 1.0],
    [-1, 0, 0, 1],
    plan,
    weights,
    limits=tiny_oracle_limits(),
  )
  assert_eq(padded.output().length(), 2)
  let invalid : Array[
    (Array[Int], @glm_hybrid_reference.GlmHybridReferenceError),
  ] = [
    ([-1, 1, 0, 1], IndexOutOfBounds),
    ([-1, 0, 0, 0], DuplicateIndex),
    ([-1, -1, 0, 1], IndexOutOfBounds),
  ]
  for fixture in invalid {
    let (selected, expected) = fixture
    try
      @glm_hybrid_reference.execute_tiny_hybrid_block(
        [1.0, 1.0],
        selected,
        plan,
        weights,
        limits=tiny_oracle_limits(),
      )
    catch {
      error => assert_eq(error, expected)
    } noraise {
      _ => fail("invalid selected indices admitted")
    }
  }
  try
    @glm_hybrid_reference.execute_tiny_hybrid_block(
      [@double.not_a_number, 1.0],
      [-1, 0, 0, 1],
      plan,
      weights,
      limits=tiny_oracle_limits(),
    )
  catch {
    NonFiniteInput => ()
    _ => fail("unexpected NaN failure")
  } noraise {
    _ => fail("NaN block input admitted")
  }
  try
    @glm_hybrid_reference.execute_tiny_hybrid_block(
      [1.0],
      [-1, 0, 0, 1],
      plan,
      weights,
      limits=tiny_oracle_limits(),
    )
  catch {
    InvalidShape => ()
    _ => fail("unexpected shape failure")
  } noraise {
    _ => fail("short block input admitted")
  }
  try
    @glm_hybrid_reference.GlmTinyVisionBlockPlan::new(
      patch_count=4,
      head_count=1,
      head_dimension=1,
      intermediate_size=1,
      merge_size=3,
      project_width=1,
      rms_epsilon=1.0,
      limits=tiny_oracle_limits(),
    )
  catch {
    InvalidVisionGeometry => ()
    _ => fail("unexpected vision bounds failure")
  } noraise {
    _ => fail("invalid vision merge geometry admitted")
  }
  let vision_plan = @glm_hybrid_reference.GlmTinyVisionBlockPlan::new(
    patch_count=4,
    head_count=1,
    head_dimension=1,
    intermediate_size=1,
    merge_size=2,
    project_width=1,
    rms_epsilon=1.0,
    limits=tiny_oracle_limits(),
  )
  let vision_weights = @glm_hybrid_reference.GlmTinyVisionBlockWeights::new(
    attention_norm=[1.0],
    query=[1.0],
    key=[1.0],
    value=[1.0],
    attention_output=[1.0],
    ffn_norm=[1.0],
    gate=[1.0],
    up=[1.0],
    down=[1.0],
    merge=[1.0, 1.0, 1.0, 1.0],
    merge_bias=[0.0],
    project=[1.0],
    project_bias=[0.0],
  )
  try
    @glm_hybrid_reference.execute_tiny_vision_block(
      [@double.not_a_number, 1.0, 1.0, 1.0],
      vision_plan,
      vision_weights,
      limits=tiny_oracle_limits(),
    )
  catch {
    NonFiniteInput => ()
    _ => fail("unexpected vision NaN failure")
  } noraise {
    _ => fail("NaN vision input admitted")
  }
}
