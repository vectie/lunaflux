///|
fn fixture_operation(
  id : Int,
  kind : @plan.OperationKind,
  inputs : ArrayView[@plan.ValueRef],
  shape : @plan.OperationShape,
  capability : @plan.KernelCapabilityId,
) -> @plan.PlanOperation raise {
  @plan.PlanOperation::new_stateless(
    id=@plan.OperationId::new(id),
    kind~,
    inputs~,
    outputs=[@plan.ActivationRef::new(id)],
    shape~,
    required_capability=capability,
    workspace_bytes=0L,
  )
}

///|
fn fixture_plan(include_head : Bool) -> @plan.ModelPlan raise {
  let operations = [
    fixture_operation(
      0,
      Embedding,
      [ModelInput(TokenIds), Tensor(@plan.TensorRef::new(0))],
      EmbeddingShape(vocabulary_size=16, width=4),
      @plan.KernelCapabilityId::embedding_lookup(),
    ),
  ]
  if include_head {
    operations.push(
      fixture_operation(
        1,
        LanguageModelHead,
        [
          Activation(@plan.ActivationRef::new(0)),
          Tensor(@plan.TensorRef::new(1)),
        ],
        ProjectionShape(input_width=4, output_width=16),
        @plan.KernelCapabilityId::dense_projection(),
      ),
    )
  }
  @plan.ModelPlan::new_stateless(
    content_digest=@spec.ContentDigest::from_sha256("1".repeat(64)),
    limits=@plan.ModelPlanLimits::production(),
    shape_constraints=@plan.ShapeConstraints::new(
      max_batch_rows=2,
      max_sequence_tokens=4,
      vocabulary_size=16,
      hidden_size=4,
      intermediate_size=8,
      attention_head_count=2,
      kv_head_count=1,
      head_dimension=2,
    ),
    kv_cache_geometry=@plan.KvCacheGeometry::new(
      layer_count=1,
      kv_head_count=1,
      head_dimension=2,
      dtype=BFloat16,
      layout=LayerMajorSplitKeyValueV1,
    ),
    workspace_bounds=@plan.WorkspaceBounds::new(max_temporary_bytes=0L),
    operations~,
    final_output=@plan.ActivationRef::new(operations.length() - 1),
    numeric_schema=@plan_test_fixture.synthetic_bf16_numeric_schema(operations),
  )
}

///|
test "BF16 language-model logits admit host-owned sampling without an artifact" {
  let plan = fixture_plan(true)
  let boundary = @luna_cuda_sampling_aot.admit_host_sampling_boundary(plan)
  assert_true(boundary.owner() == HostCompletionOwner)
  assert_eq(boundary.producer().as_int(), 1)
  assert_true(boundary.final_output() == plan.final_output())
  assert_eq(boundary.vocabulary_size(), 16)
  assert_eq(boundary.logits_row_bytes(), 32L)
  assert_false(boundary.requires_cuda_artifact())
}

///|
test "a hidden-state terminal rejects sampling-kernel fabrication" {
  let plan = fixture_plan(false)
  try @luna_cuda_sampling_aot.admit_host_sampling_boundary(plan) catch {
    Unsupported(FinalOutputKind) => ()
    _ => fail("unexpected sampling-boundary error")
  } noraise {
    _ => fail("non-logits terminal was admitted for host sampling")
  }
}

///|
fn sampling_test_target() -> @catalog.DeviceTarget {
  try! @catalog.DeviceTarget::new(
    compute_major=9,
    compute_minor=0,
    supports_bf16=true,
    supports_cublas_lt=true,
  )
}

///|
fn sampling_test_compiler(
  reassociate? : Bool = false,
) -> @luna_specialization.LunaCompilerPolicy {
  try! @luna_specialization.LunaCompilerPolicy::new(
    identity=@luna_specialization.LunaCompilerIdentity::new(
      toolchain_digest_sha256="2".repeat(64),
      major=13,
      minor=0,
      patch=0,
    ),
    optimization_level=3,
    fused_multiply_add=false,
    reassociate~,
    max_registers=64,
  )
}

///|
fn sampling_test_limits() -> GreedySamplingAotLimits {
  try! GreedySamplingAotLimits::new(
    max_source_bytes=16384,
    max_recipe_bytes=4096,
    max_module_bytes=65536L,
  )
}

///|
test "greedy AOT lowering is closed, bounded, and deterministic" {
  let boundary = @luna_cuda_sampling_aot.admit_host_sampling_boundary(
    fixture_plan(true),
  )
  let first = lower_greedy_sampling_cuda_aot(
    boundary,
    sampling_test_target(),
    sampling_test_compiler(),
    2,
    sampling_test_limits(),
  )
  let second = lower_greedy_sampling_cuda_aot(
    boundary,
    sampling_test_target(),
    sampling_test_compiler(),
    2,
    sampling_test_limits(),
  )
  assert_eq(first.source_bytes(), second.source_bytes())
  assert_eq(first.source_sha256(), second.source_sha256())
  assert_eq(first.recipe_sha256(), second.recipe_sha256())
  let source = @utf8.decode(first.source_bytes())
  assert_true(source.contains("const int32_t *counts"))
  assert_true(source.contains("const int32_t *row_offsets"))
  assert_true(source.contains("const __nv_bfloat16 *logits"))
  assert_true(source.contains("int32_t *results"))
  assert_true(source.contains("value == best && token < best_token"))
  assert_true(source.contains("!isfinite(value)"))
}

///|
test "greedy AOT rejects reassociation and nondeterministic output" {
  let boundary = @luna_cuda_sampling_aot.admit_host_sampling_boundary(
    fixture_plan(true),
  )
  try
    lower_greedy_sampling_cuda_aot(
      boundary,
      sampling_test_target(),
      sampling_test_compiler(reassociate=true),
      2,
      sampling_test_limits(),
    )
  catch {
    Unsupported(ArtifactTarget) => ()
    _ => fail("unexpected compiler-policy error")
  } noraise {
    _ => fail("reassociating compiler policy was admitted")
  }
  let lowering = lower_greedy_sampling_cuda_aot(
    boundary,
    sampling_test_target(),
    sampling_test_compiler(),
    2,
    sampling_test_limits(),
  )
  let first = b"first-cubin"
  let second = b"second-cubin"
  let receipt = GreedySamplingCompileReceipt::from_offline_builder(
    source_sha256=lowering.source_sha256(),
    recipe_sha256=lowering.recipe_sha256(),
    toolchain_sha256="2".repeat(64),
    driver_sha256="3".repeat(64),
    first_module_sha256=@crypto.bytes_to_hex_string(@crypto.sha256(first)),
    second_module_sha256=@crypto.bytes_to_hex_string(@crypto.sha256(second)),
  )
  try
    admit_greedy_sampling_cuda_artifact(
      lowering,
      first,
      second,
      receipt,
      sampling_test_limits(),
    )
  catch {
    Mismatch(ArtifactDeterminism) => ()
    _ => fail("unexpected artifact-admission error")
  } noraise {
    _ => fail("nondeterministic sampling artifact was admitted")
  }
}
