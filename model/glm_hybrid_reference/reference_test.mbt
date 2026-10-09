///|
fn glm_reference_limits() -> @glm_hybrid_reference.GlmHybridReferenceLimits {
  try! @glm_hybrid_reference.GlmHybridReferenceLimits::new(
    maximum_sequence_length=16,
    maximum_head_count=8,
    maximum_head_dimension=8,
    maximum_channel_count=64,
    maximum_vision_patch_count=16,
  )
}

///|
fn advanced_reference_limits() -> @advanced_decoder_reference.ReferenceLimits {
  try! @advanced_decoder_reference.ReferenceLimits::new(
    max_values=1024,
    max_width=32,
    max_iterations=32,
  )
}

///|
fn full_reference_plan() -> @glm_hybrid_execution_plan.GlmHybridExecutionPlan {
  let metadata = @glm53_spec.Glm53FullMetadata::from_verified_content(
    @glm53_spec.Glm53FullSpec::official(),
    try! @spec.ContentDigest::from_sha256(
      "7123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
    ),
  )
  @glm53.build_full_execution_plan(metadata)
}

///|
fn flash_reference_plan() -> @glm_hybrid_execution_plan.GlmHybridExecutionPlan {
  let metadata = @glm53_spec.Glm53FlashMetadata::from_verified_content(
    @glm53_spec.Glm53FlashSpec::official(),
    try! @spec.ContentDigest::from_sha256(
      "8123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
    ),
  )
  @glm53.build_flash_execution_plan(metadata)
}

///|
fn approximately(actual : Double, expected : Double) -> Bool {
  (actual - expected).abs() < 0.00001
}

///|
test "DSA pooling scores and shared stable top-k expand exact raw pools" {
  let pooled = @glm_hybrid_reference.pool_dsa_index_keys(
    [99.0, 1.0, 3.0, 2.0, 4.0, 9.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [false, true, true, true, true, true],
    sequence_length=6,
    head_dimension=1,
    pool_size=2,
    absolute_pool_encoding=[0.0, 0.0],
    limits=glm_reference_limits(),
  )
  assert_eq(pooled.pool_count(), 2)
  assert_eq(pooled.keys(), [2.0, 3.0])
  assert_eq(pooled.raw_indices(), [1, 2, 3, 4])
  let scores = @glm_hybrid_reference.score_dsa_pools(
    [2.0],
    [1.0],
    pooled,
    [true, true],
    query_count=1,
    head_count=1,
    limits=glm_reference_limits(),
  )
  assert_eq(scores, [4.0, 6.0])
  let state = @glm_hybrid_reference.select_dsa_pools(
    [2.0],
    [1.0],
    pooled,
    [0, 1],
    head_count=1,
    top_k_pools=1,
    source_layer=2,
    key_count=6,
    limits=glm_reference_limits(),
    advanced_limits=advanced_reference_limits(),
  )
  assert_eq(state.indices(), [3, 4])
  let reused = @glm_hybrid_reference.reuse_dsa_topk(
    state,
    consumer_layer=3,
    key_count=6,
  )
  assert_eq(reused.source_layer(), 2)
  assert_eq(reused.last_consumer_layer(), 3)
}

///|
test "KDA causal convolution and recurrent delta update match hand fixtures" {
  let convolution = @glm_hybrid_reference.kda_causal_short_convolution(
    [1.0, 2.0, 3.0],
    [2.0, 3.0],
    [10.0],
    sequence_length=3,
    channel_count=1,
    kernel_size=2,
    limits=glm_reference_limits(),
  )
  assert_true(
    approximately(convolution.output()[0], 23.0 / (1.0 + @math.exp(-23.0))),
  )
  assert_true(
    approximately(convolution.output()[1], 8.0 / (1.0 + @math.exp(-8.0))),
  )
  assert_true(
    approximately(convolution.output()[2], 13.0 / (1.0 + @math.exp(-13.0))),
  )
  assert_eq(convolution.final_state(), [3.0])
  let recurrent = @glm_hybrid_reference.kda_recurrent_delta_update(
    [1.0, 1.0],
    [1.0, 1.0],
    [2.0, 3.0],
    [0.0, 0.0],
    [1.0, 1.0],
    [0.0],
    sequence_length=2,
    head_count=1,
    head_dimension=1,
    limits=glm_reference_limits(),
  )
  assert_true(approximately(recurrent.output()[0], 1.999998))
  assert_true(approximately(recurrent.output()[1], 2.999997))
  assert_true(approximately(recurrent.final_state()[0], 2.999998))
  let component_decay = @glm_hybrid_reference.kda_recurrent_delta_update(
    [1.0, 0.0],
    [1.0, 0.0],
    [0.0, 0.0],
    [0.0, -2.0],
    [0.0],
    [1.0, 2.0, 3.0, 4.0],
    sequence_length=1,
    head_count=1,
    head_dimension=2,
    limits=glm_reference_limits(),
  )
  assert_true(approximately(component_decay.final_state()[0], 1.0))
  assert_true(approximately(component_decay.final_state()[1], 2.0))
  assert_true(
    approximately(component_decay.final_state()[2], 3.0 * @math.exp(-2.0)),
  )
  assert_true(
    approximately(component_decay.final_state()[3], 4.0 * @math.exp(-2.0)),
  )
}

///|
test "KDA projection preserves QKV two-stage forget and beta seams" {
  let projected = @glm_hybrid_reference.kda_projection_control(
    [2.0, -1.0],
    [1.0, 0.0, 0.0, 1.0],
    [0.0, 1.0, 1.0, 0.0],
    [1.0, 1.0, 1.0, -1.0],
    [1.0, 2.0, -1.0, 1.0],
    [1.0, 0.0, 0.0, 1.0],
    [1.0, -1.0],
    row_count=1,
    hidden_width=2,
    head_count=1,
    head_dimension=2,
    control_rank=2,
    limits=glm_reference_limits(),
  )
  assert_eq(projected.query(), [2.0, -1.0])
  assert_eq(projected.key(), [-1.0, 2.0])
  assert_eq(projected.value(), [1.0, 3.0])
  assert_eq(projected.forget_low_rank(), [0.0, -3.0])
  assert_eq(projected.raw_forget(), [0.0, -3.0])
  assert_true(approximately(projected.beta()[0], 1.0 / (1.0 + @math.exp(-3.0))))
}

///|
test "KDA output gate preserves the explicit low-rank boundary" {
  let gate = @glm_hybrid_reference.kda_output_gate_projection(
    [2.0, -1.0],
    [1.0, 2.0, -1.0, 1.0],
    [1.0, 0.0, 0.0, 1.0],
    row_count=1,
    hidden_width=2,
    head_count=1,
    head_dimension=2,
    limits=glm_reference_limits(),
  )
  assert_eq(gate.low_rank(), [0.0, -3.0])
  assert_eq(gate.gate(), [0.0, -3.0])
}

///|
test "vision patch downsample and row projection match independent matrices" {
  let patches = @glm_hybrid_reference.vision_patch_projection(
    [1.0, 2.0, 3.0, 4.0],
    [1.0, 0.0, 0.0, 1.0],
    [1.0, -1.0],
    patch_count=2,
    patch_volume=2,
    output_width=2,
    limits=glm_reference_limits(),
  )
  assert_eq(patches.values(), [2.0, 1.0, 4.0, 3.0])
  let merged = @glm_hybrid_reference.vision_spatial_merge(
    [1.0, 2.0, 3.0, 4.0],
    [1.0, 2.0, 3.0, 4.0],
    [0.0],
    patch_count=4,
    input_width=1,
    output_width=1,
    merge_size=2,
    limits=glm_reference_limits(),
  )
  assert_eq(merged.values(), [30.0])
  let projected = @glm_hybrid_reference.vision_project_rows(
    merged,
    [2.0, 3.0],
    [1.0, -1.0],
    output_width=2,
    limits=glm_reference_limits(),
  )
  assert_eq(projected.values(), [61.0, 89.0])
}

///|
test "adapter traverses exact full and Flash layer schedules" {
  let full = @glm_hybrid_reference.build_reference_schedule(
    full_reference_plan(),
  )
  assert_eq(full.steps().length(), 78)
  assert_eq(full.dsa_compute_count(), 21)
  assert_eq(full.dsa_reuse_count(), 57)
  assert_eq(full.kda_count(), 0)
  assert_eq(full.dense_count(), 3)
  assert_eq(full.moe_count(), 75)
  assert_eq(full.steps()[3].attention(), DsaReuse(source_layer=2))
  let flash = @glm_hybrid_reference.build_reference_schedule(
    flash_reference_plan(),
  )
  assert_eq(flash.steps().length(), 45)
  assert_eq(flash.dsa_compute_count(), 11)
  assert_eq(flash.dsa_reuse_count(), 0)
  assert_eq(flash.kda_count(), 34)
  assert_eq(flash.hyper_connection_site_count(), 90)
  assert_eq(flash.vision_layer_count(), 24)
}

///|
test "GLM adapter exercises shared MoE top-k and mHC primitives" {
  let full = @glm_hybrid_reference.exercise_shared_advanced_primitives(
    full_reference_plan(),
    limits=advanced_reference_limits(),
  )
  assert_true(full.moe_output() > 0.0F)
  assert_eq(full.mixed_hyper_streams().length(), 0)
  let flash = @glm_hybrid_reference.exercise_shared_advanced_primitives(
    flash_reference_plan(),
    limits=advanced_reference_limits(),
  )
  assert_true(flash.moe_output() > 0.0F)
  assert_eq(flash.mixed_hyper_streams().length(), 4)
  let mut sum = 0.0F
  for value in flash.mixed_hyper_streams() {
    sum += value
  }
  assert_true((sum - 10.0F).abs() < 0.001F)
}

///|
test "reference rejects stale state order duplicates NaN probabilities and bounds" {
  let state = @glm_hybrid_reference.record_dsa_topk(
    source_layer=0,
    key_count=4,
    indices=[0, 1],
    limits=glm_reference_limits(),
  )
  try
    @glm_hybrid_reference.reuse_dsa_topk(state, consumer_layer=2, key_count=4)
  catch {
    InvalidLayerOrder => ()
    _ => fail("unexpected stale order failure")
  } noraise {
    _ => fail("skipped DSA consumer admitted")
  }
  try
    @glm_hybrid_reference.record_dsa_topk(
      source_layer=0,
      key_count=4,
      indices=[1, 1],
      limits=glm_reference_limits(),
    )
  catch {
    DuplicateIndex => ()
    _ => fail("unexpected duplicate index failure")
  } noraise {
    _ => fail("duplicate DSA index admitted")
  }
  try
    @glm_hybrid_reference.kda_recurrent_delta_update(
      [@double.not_a_number],
      [1.0],
      [1.0],
      [0.0],
      [1.0],
      [0.0],
      sequence_length=1,
      head_count=1,
      head_dimension=1,
      limits=glm_reference_limits(),
    )
  catch {
    NonFiniteInput => ()
    _ => fail("unexpected NaN failure")
  } noraise {
    _ => fail("NaN KDA input admitted")
  }
  try
    @glm_hybrid_reference.kda_recurrent_delta_update(
      [1.0],
      [1.0],
      [1.0],
      [0.1],
      [1.0],
      [0.0],
      sequence_length=1,
      head_count=1,
      head_dimension=1,
      limits=glm_reference_limits(),
    )
  catch {
    InvalidProbability => ()
    _ => fail("unexpected decay failure")
  } noraise {
    _ => fail("amplifying KDA decay admitted")
  }
  try
    @glm_hybrid_reference.vision_patch_projection(
      [1.0],
      [1.0],
      [0.0],
      patch_count=17,
      patch_volume=1,
      output_width=1,
      limits=glm_reference_limits(),
    )
  catch {
    BoundsExceeded => ()
    _ => fail("unexpected vision bound failure")
  } noraise {
    _ => fail("oversized vision fixture admitted")
  }
}
