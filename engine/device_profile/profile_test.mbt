///|
test "one-layer Llama prefill derives every exact dispatch view" {
  let static_plan = profile_static_plan()
  let memory_plan = profile_memory_plan(static_plan~)
  let shape = @device_profile.ExactExecutionShape::full_prefill(
    batch_rows=1,
    sequence_tokens=4,
  )
  let profile = @device_profile.build(static_plan, memory_plan, shape)
  assert_eq(profile.identity(), static_plan.identity())
  assert_eq(profile.target(), static_plan.target())
  assert_eq(profile.shape(), shape)
  assert_eq(shape.phase().as_stable_int(), 0)
  assert_eq(shape.query_tokens_per_row(), 4)
  assert_eq(shape.context_tokens(), 4)
  assert_eq(shape.token_rows(), 4L)
  assert_eq(profile.operation_count(), 12)
  assert_eq(profile.region_count(), 26)
  assert_eq(profile.token_staging().id().as_int(), 0)
  assert_eq(profile.token_staging().space(), TokenInput)
  assert_eq(profile.token_staging().logical_bytes(), 16L)
  assert_eq(profile.token_staging().reserved_bytes(), 64L)
  assert_eq(profile.token_staging_capacity_bytes(), 64L)
  for index in 0..<profile.operation_count() {
    let operation = profile.operation(@plan.OperationId::new(index)).unwrap()
    assert_eq(operation.operation_id().as_int(), index)
    assert_eq(operation.kernel_shape(), shape.kernel_shape())
    assert_eq(operation.outputs().length(), 1)
    assert_eq(operation.outputs()[0].token_rows(), 4L)
    assert_eq(operation.workspace().logical_bytes(), 32L)
    assert_true(
      operation.workspace().logical_bytes() <=
      operation.workspace().reserved_bytes(),
    )
  }
  let embedding = profile.operation(@plan.OperationId::new(0)).unwrap()
  assert_eq(embedding.inputs().length(), 2)
  guard embedding.inputs()[0] is TokenIds(token_ids) else {
    fail("expected exact token input")
  }
  assert_eq(token_ids, profile.token_staging())
  guard embedding.inputs()[1] is Weight(reference, embedding_weight) else {
    fail("expected embedding weight")
  }
  assert_eq(reference.as_int(), 0)
  assert_eq(embedding_weight.id().as_int(), 1)
  assert_eq(embedding_weight.logical_bytes(), 64L)
  assert_eq(embedding.outputs()[0].width(), 4L)
  assert_eq(embedding.outputs()[0].region().logical_bytes(), 32L)
  assert_eq(embedding.outputs()[0].region().id().as_int(), 13)
  let qkv = profile.operation(@plan.OperationId::new(2)).unwrap()
  assert_eq(qkv.inputs().length(), 4)
  assert_eq(qkv.outputs()[0].width(), 8L)
  assert_eq(qkv.outputs()[0].region().logical_bytes(), 64L)
  let mlp = profile.operation(@plan.OperationId::new(8)).unwrap()
  assert_eq(mlp.inputs().length(), 4)
  assert_eq(mlp.outputs()[0].width(), 4L)
  assert_eq(mlp.outputs()[0].region().logical_bytes(), 32L)
  let final_output = profile.final_output()
  assert_eq(final_output.semantics(), LanguageModelLogits)
  assert_eq(final_output.value().reference().as_int(), 11)
  assert_eq(final_output.value().width(), 8L)
  assert_eq(final_output.value().region().logical_bytes(), 64L)
  assert_eq(final_output.row_bytes(), 16L)
  assert_eq(final_output.selected_rows().length(), 1)
  assert_eq(
    final_output.selected_rows()[0].offset_bytes(),
    final_output.value().region().offset_bytes() + 48L,
  )
  assert_eq(final_output.selected_rows()[0].byte_length(), 16L)
  assert_eq(
    final_output.selected_rows()[0].backing_region(),
    final_output.value().region().id(),
  )
  let workspace_id = 25
  assert_eq(
    profile
    .region(@device_profile.ProfileRegionId::from_stable_int(workspace_id))
    .unwrap()
    .space(),
    ActivationWorkspaceArena,
  )
}

///|
test "full recompute retains the complete stateless sequence" {
  let static_plan = profile_static_plan()
  let memory_plan = profile_memory_plan(static_plan~)
  let shape = @device_profile.ExactExecutionShape::full_recompute(
    batch_rows=1,
    sequence_tokens=7,
  )
  let profile = @device_profile.build(static_plan, memory_plan, shape)
  assert_eq(shape.phase().as_stable_int(), 1)
  assert_eq(shape.query_tokens_per_row(), 7)
  assert_eq(shape.context_tokens(), 7)
  assert_eq(shape.token_rows(), 7L)
  assert_eq(profile.token_staging().logical_bytes(), 28L)
  assert_eq(
    profile.operation(@plan.OperationId::new(2)).unwrap().outputs()[0]
    .region()
    .logical_bytes(),
    112L,
  )
  assert_eq(profile.final_output().value().region().logical_bytes(), 112L)
  assert_eq(
    profile.final_output().selected_rows()[0].offset_bytes(),
    profile.final_output().value().region().offset_bytes() + 96L,
  )
}

///|
test "shape and plan envelope mismatches fail closed" {
  let static_plan = profile_static_plan()
  let memory_plan = profile_memory_plan(static_plan~)
  for
    shape in [
      @device_profile.ExactExecutionShape::full_prefill(
        batch_rows=2,
        sequence_tokens=1,
      ),
      @device_profile.ExactExecutionShape::full_prefill(
        batch_rows=1,
        sequence_tokens=17,
      ),
      @device_profile.ExactExecutionShape::full_recompute(
        batch_rows=1,
        sequence_tokens=17,
      ),
    ] {
    try @device_profile.build(static_plan, memory_plan, shape) catch {
      InvalidShape(BatchExceedsEnvelope) | InvalidShape(QueryExceedsEnvelope) =>
        ()
      _ => fail("unexpected profile-envelope error")
    } noraise {
      _ => fail("expected exact shape outside the static envelope")
    }
  }
  let invalid_shapes : Array[(Int, Int, @device_profile.ProfileShapeRule)] = [
    (0, 1, NonPositiveBatchRows),
    (1, 0, NonPositiveQueryTokens),
  ]
  for item in invalid_shapes {
    let (batch_rows, sequence_tokens, expected) = item
    try
      @device_profile.ExactExecutionShape::full_prefill(
        batch_rows~,
        sequence_tokens~,
      )
    catch {
      InvalidShape(rule) => assert_eq(rule, expected)
      _ => fail("unexpected non-positive shape error")
    } noraise {
      _ => fail("expected invalid exact shape")
    }
  }
  let other_target = profile_test_target(compute_major=9)
  let other_static = profile_static_plan(target=other_target)
  try
    @device_profile.build(
      other_static,
      memory_plan,
      @device_profile.ExactExecutionShape::full_prefill(
        batch_rows=1,
        sequence_tokens=1,
      ),
    )
  catch {
    InvalidPlan(Target) => ()
    _ => fail("unexpected target mismatch error")
  } noraise {
    _ => fail("expected target mismatch")
  }
  let other_model = profile_test_model(
    digest="abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789",
  )
  let other_semantic = @llama.build(other_model)
  let other_identity_static = profile_static_plan(
    semantic=other_semantic,
    layout=profile_test_layout(model=other_model),
  )
  try
    @device_profile.build(
      other_identity_static,
      memory_plan,
      @device_profile.ExactExecutionShape::full_prefill(
        batch_rows=1,
        sequence_tokens=1,
      ),
    )
  catch {
    InvalidPlan(Identity) => ()
    _ => fail("unexpected identity mismatch error")
  } noraise {
    _ => fail("expected identity mismatch")
  }
}

///|
test "maximum memory envelope is not substituted for another static plan" {
  let semantic_small = profile_embedding_plan(1, 8)
  let semantic_large = profile_embedding_plan(1, 16)
  let static_small = profile_static_plan(semantic=semantic_small)
  let static_large = profile_static_plan(semantic=semantic_large)
  let memory_large = profile_memory_plan(static_plan=static_large)
  try
    @device_profile.build(
      static_small,
      memory_large,
      @device_profile.ExactExecutionShape::full_prefill(
        batch_rows=1,
        sequence_tokens=4,
      ),
    )
  catch {
    InvalidPlan(MaximumTokenRows) => ()
    _ => fail("unexpected maximum-envelope mismatch error")
  } noraise {
    _ => fail("expected exact static/memory envelope mismatch")
  }
}

///|
test "multi-batch terminal rows are precomputed without assuming contiguity" {
  let semantic = profile_embedding_plan(2, 8)
  let static_plan = profile_static_plan(semantic~)
  let memory_plan = profile_memory_plan(static_plan~)
  let profile = @device_profile.build(
    static_plan,
    memory_plan,
    @device_profile.ExactExecutionShape::full_prefill(
      batch_rows=2,
      sequence_tokens=3,
    ),
  )
  let output = profile.final_output()
  assert_eq(output.semantics(), ActivationOutput)
  assert_eq(output.value().token_rows(), 6L)
  assert_eq(output.value().region().logical_bytes(), 48L)
  assert_eq(output.row_bytes(), 8L)
  assert_eq(output.selected_rows().length(), 2)
  let base = output.value().region().offset_bytes()
  assert_eq(output.selected_rows()[0].offset_bytes(), base + 16L)
  assert_eq(output.selected_rows()[1].offset_bytes(), base + 40L)
}

///|
test "zero workspace is a valid omitted-operand view" {
  let semantic = profile_embedding_plan(1, 8)
  let static_plan = profile_static_plan(semantic~, workspace_bytes=0L)
  let memory_plan = profile_memory_plan(static_plan~)
  let profile = @device_profile.build(
    static_plan,
    memory_plan,
    @device_profile.ExactExecutionShape::full_prefill(
      batch_rows=1,
      sequence_tokens=2,
    ),
  )
  let workspace = profile
    .operation(@plan.OperationId::new(0))
    .unwrap()
    .workspace()
  assert_eq(workspace.logical_bytes(), 0L)
  assert_eq(workspace.reserved_bytes(), 0L)
  assert_eq(workspace.end_offset_bytes(), workspace.offset_bytes())
}
