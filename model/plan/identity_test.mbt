///|
fn identity_test_content(hex : String) -> @spec.ContentDigest {
  try! @spec.ContentDigest::from_sha256(hex)
}

///|
fn identity_test_stateless_plan(
  content : @spec.ContentDigest,
  workspace_bytes? : Int64 = 4096L,
  limits? : @plan.ModelPlanLimits = @plan.ModelPlanLimits::production(),
) -> @plan.ModelPlan raise @plan.PlanValidationError {
  let operation = @plan.PlanOperation::new_stateless(
    id=@plan.OperationId::new(0),
    kind=Embedding,
    inputs=[ModelInput(TokenIds), Tensor(@plan.TensorRef::new(0))],
    outputs=[@plan.ActivationRef::new(0)],
    shape=EmbeddingShape(vocabulary_size=128, width=16),
    required_capability=@plan.KernelCapabilityId::embedding_lookup(),
    workspace_bytes=0L,
  )
  let operations = [operation]
  @plan.ModelPlan::new_stateless(
    content_digest=content,
    limits~,
    shape_constraints=plan_test_constraints(),
    kv_cache_geometry=plan_test_kv_geometry(),
    workspace_bounds=@plan.WorkspaceBounds::new(
      max_temporary_bytes=workspace_bytes,
    ),
    operations~,
    final_output=@plan.ActivationRef::new(0),
    numeric_schema=plan_test_numeric_schema(operations),
  )
}

///|
test "model plan mints the pinned minimal stateless canonical identity" {
  let content = identity_test_content(
    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
  )
  let first = identity_test_stateless_plan(content)
  let second = identity_test_stateless_plan(content)
  inspect(
    first.identity().plan().as_hex(),
    content="18ac044a95b6c61e3fbbb80531ee4a1828d88525b041eea749ddd933c4b208b2",
  )
  assert_eq(first.identity(), second.identity())
}

///|
test "content and plan digests have independent authority" {
  let first = identity_test_stateless_plan(
    identity_test_content(
      "1111111111111111111111111111111111111111111111111111111111111111",
    ),
  )
  let second = identity_test_stateless_plan(
    identity_test_content(
      "2222222222222222222222222222222222222222222222222222222222222222",
    ),
  )
  assert_eq(first.identity().plan(), second.identity().plan())
  assert_false(first.identity() == second.identity())
  let workspace_mutation = identity_test_stateless_plan(
    first.identity().content(),
    workspace_bytes=8192L,
  )
  assert_false(first.identity().plan() == workspace_mutation.identity().plan())
}

///|
test "model plan limits reject hostile inputs before graph ownership" {
  let content = identity_test_content(
    "3333333333333333333333333333333333333333333333333333333333333333",
  )
  let tiny_bytes = @plan.ModelPlanLimits::new(
    max_operations=1,
    max_numeric_tensors=1,
    max_total_value_inputs=2,
    max_value_inputs_per_operation=2,
    max_runtime_inputs_per_operation=1,
    max_outputs_per_operation=1,
    max_canonical_bytes=1,
  )
  try identity_test_stateless_plan(content, limits=tiny_bytes) catch {
    LimitExceeded(CanonicalBytes) => ()
    _ => fail("unexpected canonical limit failure")
  } noraise {
    _ => fail("tiny canonical plan limit was ignored")
  }
  let operation = @plan.PlanOperation::new_stateless(
    id=@plan.OperationId::new(0),
    kind=Embedding,
    inputs=[ModelInput(TokenIds), Tensor(@plan.TensorRef::new(0))],
    outputs=[@plan.ActivationRef::new(0)],
    shape=EmbeddingShape(vocabulary_size=128, width=16),
    required_capability=@plan.KernelCapabilityId::embedding_lookup(),
    workspace_bytes=0L,
  )
  let operations = [operation, operation]
  let one_operation = @plan.ModelPlanLimits::new(
    max_operations=1,
    max_numeric_tensors=1,
    max_total_value_inputs=2,
    max_value_inputs_per_operation=2,
    max_runtime_inputs_per_operation=1,
    max_outputs_per_operation=1,
    max_canonical_bytes=4096,
  )
  try
    @plan.ModelPlan::new_stateless(
      content_digest=content,
      limits=one_operation,
      shape_constraints=plan_test_constraints(),
      kv_cache_geometry=plan_test_kv_geometry(),
      workspace_bounds=@plan.WorkspaceBounds::new(max_temporary_bytes=0L),
      operations~,
      final_output=@plan.ActivationRef::new(0),
      numeric_schema=plan_test_numeric_schema([operation]),
    )
  catch {
    LimitExceeded(Operations) => ()
    _ => fail("unexpected operation limit failure")
  } noraise {
    _ => fail("operation-count limit was ignored")
  }
}

///|
test "model plan limit factory is hard-bounded and coherent" {
  try
    @plan.ModelPlanLimits::new(
      max_operations=4097,
      max_numeric_tensors=1,
      max_total_value_inputs=1,
      max_value_inputs_per_operation=1,
      max_runtime_inputs_per_operation=1,
      max_outputs_per_operation=1,
      max_canonical_bytes=1,
    )
  catch {
    InvalidLimit(Operations) => ()
    _ => fail("unexpected hard-ceiling failure")
  } noraise {
    _ => fail("hard operation ceiling was bypassed")
  }
  try
    @plan.ModelPlanLimits::new(
      max_operations=1,
      max_numeric_tensors=1,
      max_total_value_inputs=1,
      max_value_inputs_per_operation=2,
      max_runtime_inputs_per_operation=1,
      max_outputs_per_operation=1,
      max_canonical_bytes=1,
    )
  catch {
    InvalidLimit(IncoherentLimits) => ()
    _ => fail("unexpected limit-coherence failure")
  } noraise {
    _ => fail("incoherent plan limits were admitted")
  }
}
