///|
fn plan_test_identity() -> @spec.ModelIdentity {
  let content = try! @spec.ContentDigest::from_sha256(
    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
  )
  let digest = try! @spec.PlanDigest::from_sha256(
    "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789",
  )
  @spec.ModelIdentity::new(content, digest)
}

///|
fn plan_test_constraints() -> @plan.ShapeConstraints {
  try! @plan.ShapeConstraints::new(
    max_batch_rows=1,
    max_sequence_tokens=64,
    vocabulary_size=128,
    hidden_size=16,
    intermediate_size=32,
    attention_head_count=4,
    kv_head_count=2,
    head_dimension=4,
  )
}

///|
fn plan_test_kv_geometry(
  kv_head_count? : Int = 2,
  head_dimension? : Int = 4,
) -> @plan.KvCacheGeometry {
  try! @plan.KvCacheGeometry::new(
    layer_count=1,
    kv_head_count~,
    head_dimension~,
    dtype=BFloat16,
    layout=LayerMajorSplitKeyValueV1,
  )
}

///|
test "KV layer identities are nonnegative semantic scalars" {
  assert_eq(try! @plan.KvLayerId::new(3).as_int(), 3)
  try @plan.KvLayerId::new(-1) catch {
    InvalidIdentifier(KvLayer) => ()
    _ => fail("unexpected KV layer identity failure")
  } noraise {
    _ => fail("expected negative KV layer rejection")
  }
}

///|
test "stateless operation construction rejects KV-bearing operation kinds" {
  let kv_kinds : Array[@plan.OperationKind] = [RotaryEmbedding, CausalAttention]
  for kind in kv_kinds {
    try
      @plan.PlanOperation::new_stateless(
        id=@plan.OperationId::new(0),
        kind~,
        inputs=[Activation(@plan.ActivationRef::new(0))],
        outputs=[@plan.ActivationRef::new(1)],
        shape=AttentionShape(query_heads=4, kv_heads=2, head_dimension=4),
        required_capability=@plan.KernelCapabilityId::causal_attention(),
        workspace_bytes=0L,
      )
    catch {
      InvalidGraph(KvExecutionContract) => ()
      _ => fail("unexpected stateless KV operation failure")
    } noraise {
      _ => fail("expected stateless KV operation rejection")
    }
  }
}

///|
test "stateless plan construction rejects explicitly layered attention" {
  let attention = @plan.PlanOperation::new_with_execution(
    id=@plan.OperationId::new(0),
    kind=CausalAttention,
    inputs=[Activation(@plan.ActivationRef::new(0))],
    outputs=[@plan.ActivationRef::new(1)],
    shape=AttentionShape(query_heads=4, kv_heads=2, head_dimension=4),
    required_capability=@plan.KernelCapabilityId::causal_attention(),
    workspace_bytes=0L,
    kv_layer=@plan.KvLayerId::new(0),
    runtime_inputs=[],
    persistent_state=NoPersistentState,
  )
  try
    @plan.ModelPlan::new_stateless(
      identity=plan_test_identity(),
      shape_constraints=plan_test_constraints(),
      kv_cache_geometry=plan_test_kv_geometry(),
      workspace_bounds=@plan.WorkspaceBounds::new(max_temporary_bytes=0L),
      operations=[attention],
      final_output=@plan.ActivationRef::new(1),
    )
  catch {
    InvalidGraph(KvExecutionContract) => ()
    _ => fail("unexpected stateless plan KV failure")
  } noraise {
    _ => fail("expected stateless plan KV rejection")
  }
}

///|
fn embedding_operation(
  first_input : @plan.ValueRef,
  capability : @plan.KernelCapabilityId,
) -> @plan.PlanOperation raise {
  @plan.PlanOperation::new_stateless(
    id=@plan.OperationId::new(0),
    kind=Embedding,
    inputs=[first_input, Tensor(@plan.TensorRef::new(0))],
    outputs=[@plan.ActivationRef::new(0)],
    shape=EmbeddingShape(vocabulary_size=128, width=16),
    required_capability=capability,
    workspace_bytes=0L,
  )
}

///|
test "model plan copies construction arrays into immutable ordered storage" {
  let operation = embedding_operation(
    ModelInput(TokenIds),
    @plan.KernelCapabilityId::embedding_lookup(),
  )
  let operations = [operation]
  let model_plan = @plan.ModelPlan::new_stateless(
    identity=plan_test_identity(),
    shape_constraints=plan_test_constraints(),
    kv_cache_geometry=plan_test_kv_geometry(),
    workspace_bounds=@plan.WorkspaceBounds::new(max_temporary_bytes=4096L),
    operations~,
    final_output=@plan.ActivationRef::new(0),
  )
  operations.push(operation)
  assert_eq(model_plan.operations().length(), 1)
  assert_eq(model_plan.required_capabilities().length(), 1)
  assert_eq(model_plan.final_output().as_int(), 0)
  assert_eq(model_plan.kv_cache_geometry(), plan_test_kv_geometry())
}

///|
test "model plan rejects KV geometry inconsistent with semantic dimensions" {
  let operation = embedding_operation(
    ModelInput(TokenIds),
    @plan.KernelCapabilityId::embedding_lookup(),
  )
  try
    @plan.ModelPlan::new_stateless(
      identity=plan_test_identity(),
      shape_constraints=plan_test_constraints(),
      kv_cache_geometry=plan_test_kv_geometry(kv_head_count=1),
      workspace_bounds=@plan.WorkspaceBounds::new(max_temporary_bytes=4096L),
      operations=[operation],
      final_output=@plan.ActivationRef::new(0),
    )
  catch {
    InvalidShape(KvCacheShapeMismatch) => ()
    _ => fail("unexpected KV shape-consistency error")
  } noraise {
    _ => fail("expected inconsistent KV geometry rejection")
  }
}

///|
test "invalid shape relationships fail before plan construction" {
  try
    @plan.ShapeConstraints::new(
      max_batch_rows=1,
      max_sequence_tokens=64,
      vocabulary_size=128,
      hidden_size=17,
      intermediate_size=32,
      attention_head_count=4,
      kv_head_count=2,
      head_dimension=4,
    )
  catch {
    InvalidShape(HiddenHeadRelationship) => ()
    _ => fail("unexpected validation error")
  } noraise {
    _ => fail("expected invalid hidden/head relationship")
  }
}

///|
test "model plan rejects forward activation references" {
  let operation = @plan.PlanOperation::new_stateless(
    id=@plan.OperationId::new(0),
    kind=RmsNorm,
    inputs=[
      Activation(@plan.ActivationRef::new(9)),
      Tensor(@plan.TensorRef::new(0)),
    ],
    outputs=[@plan.ActivationRef::new(0)],
    shape=NormalizationShape(width=16, epsilon=0.000001),
    required_capability=@plan.KernelCapabilityId::rms_norm(),
    workspace_bytes=0L,
  )
  try
    @plan.ModelPlan::new_stateless(
      identity=plan_test_identity(),
      shape_constraints=plan_test_constraints(),
      kv_cache_geometry=plan_test_kv_geometry(),
      workspace_bounds=@plan.WorkspaceBounds::new(max_temporary_bytes=4096L),
      operations=[operation],
      final_output=@plan.ActivationRef::new(0),
    )
  catch {
    InvalidGraph(MissingActivationInput) => ()
    _ => fail("unexpected validation error")
  } noraise {
    _ => fail("expected missing activation input")
  }
}

///|
test "operation kind requires its stable semantic capability" {
  let operation = embedding_operation(
    ModelInput(TokenIds),
    @plan.KernelCapabilityId::rms_norm(),
  )
  try
    @plan.ModelPlan::new_stateless(
      identity=plan_test_identity(),
      shape_constraints=plan_test_constraints(),
      kv_cache_geometry=plan_test_kv_geometry(),
      workspace_bounds=@plan.WorkspaceBounds::new(max_temporary_bytes=4096L),
      operations=[operation],
      final_output=@plan.ActivationRef::new(0),
    )
  catch {
    InvalidGraph(OperationCapabilityMismatch) => ()
    _ => fail("unexpected validation error")
  } noraise {
    _ => fail("expected operation capability mismatch")
  }
}

///|
test "operation kinds enforce exact semantic input roles" {
  let operation = embedding_operation(
    Tensor(@plan.TensorRef::new(1)),
    @plan.KernelCapabilityId::embedding_lookup(),
  )
  try
    @plan.ModelPlan::new_stateless(
      identity=plan_test_identity(),
      shape_constraints=plan_test_constraints(),
      kv_cache_geometry=plan_test_kv_geometry(),
      workspace_bounds=@plan.WorkspaceBounds::new(max_temporary_bytes=4096L),
      operations=[operation],
      final_output=@plan.ActivationRef::new(0),
    )
  catch {
    InvalidGraph(OperationInputRoles) => ()
    _ => fail("unexpected validation error")
  } noraise {
    _ => fail("expected operation input-role mismatch")
  }
}
