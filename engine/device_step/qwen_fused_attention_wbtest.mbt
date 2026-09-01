///|
fn qwen_fused_test_plan() -> @plan.ModelPlan raise {
  let spec = @qwen3_spec.Qwen3ModelSpec::new(
    vocabulary_size=151936,
    hidden_size=1024,
    intermediate_size=3072,
    layer_count=28,
    attention_head_count=16,
    kv_head_count=8,
    head_dimension=128,
    context_length=40960,
    rope_theta=1000000.0,
    weight_dtype=BFloat16,
    kv_dtype=BFloat16,
    tied_embeddings=true,
  )
  let metadata = @qwen3_spec.Qwen3ModelMetadata::from_verified_content(
    spec,
    @spec.ContentDigest::from_sha256(
      "3333333333333333333333333333333333333333333333333333333333333333",
    ),
  )
  @qwen3.build_paged(metadata, batch=@qwen3.Qwen3PagedBatchEnvelope::new(3))
}

///|
fn qwen_fused_test_blueprint(
  qwen_plan : @plan.ModelPlan,
) -> PagedKvExecutionBlueprint raise {
  let metadata = blueprint_metadata()
  let plan = blueprint_model_plan(metadata)
  let layout = blueprint_layout(plan)
  let target = blueprint_target()
  let static_plan = full_static_plan(metadata, plan, layout, target)
  let memory = blueprint_memory_plan(metadata, plan, layout, target)
  let contracts = full_contract_set(plan, static_plan, memory, layout, target)
  let base = admit_paged_graph_execution_blueprint(
    plan,
    blueprint_limits(),
    layout,
    static_plan,
    memory,
    blueprint_profile().id(),
    contracts,
    blueprint_artifacts(contracts),
  )
  let template = base.steps()[0]
  let steps : Array[PagedKvExecutionStep] = []
  for operation in qwen_plan.operations() {
    steps.push({
      ..template,
      operation_id: operation.id(),
      operation_kind: operation.kind(),
      kv_layer: operation.kv_layer(),
    })
  }
  {
    ..base,
    identity: qwen_plan.identity(),
    steps: ReadOnlyArray::from_array(steps),
  }
}

///|
fn qwen_fused_test_runtime(
  plan : @plan.ModelPlan,
  blueprint : PagedKvExecutionBlueprint,
  include_residual? : Bool = false,
  include_readonly? : Bool = true,
  ingress_grid_y? : Int = 32,
) -> @luna_fused_artifact_admission.ApprovedReusableFusedRuntimeBundle raise {
  let ingress = @luna_fused_artifact_admission.ReusableFusedRuntimeModuleInput::new(
    abi=QwenQkNormPositionedRopePagedKvWriteProductionV1,
    first_operation=3,
    operation_count=3,
    function_symbol="qwen_postprocess",
    launch=@launch_contract.AotLaunchDimensions::new(
      grid_x=blueprint.profile().max_query_tokens(),
      grid_y=ingress_grid_y,
      grid_z=1,
      block_x=128,
      block_y=1,
      block_z=1,
      shared_memory_bytes=0,
    ),
    module_bytes=b"qwen-postprocess-module",
  )
  let modules : Array[
    @luna_fused_artifact_admission.ReusableFusedRuntimeModuleInput,
  ] = []
  if include_residual {
    modules.push(
      @luna_fused_artifact_admission.ReusableFusedRuntimeModuleInput::new(
        abi=ResidualRmsNormProductionV2,
        first_operation=7,
        operation_count=2,
        function_symbol="residual_rmsnorm",
        launch=@launch_contract.AotLaunchDimensions::new(
          grid_x=blueprint.profile().max_query_tokens(),
          grid_y=1,
          grid_z=1,
          block_x=128,
          block_y=1,
          block_z=1,
          shared_memory_bytes=0,
        ),
        module_bytes=b"residual-module",
      ),
    )
  }
  modules.push(ingress)
  if include_readonly {
    modules.push(
      @luna_fused_artifact_admission.ReusableFusedRuntimeModuleInput::new(
        abi=QwenPagedAttentionReadOnlyProductionV2,
        first_operation=5,
        operation_count=1,
        function_symbol="qwen_readonly",
        launch=@launch_contract.AotLaunchDimensions::new(
          grid_x=Int::min(blueprint.profile().max_query_tokens(), 32),
          grid_y=16,
          grid_z=1,
          block_x=32,
          block_y=1,
          block_z=1,
          shared_memory_bytes=0,
        ),
        module_bytes=b"qwen-readonly-module",
      ),
    )
  }
  let canonical = @luna_fused_artifact_admission.export_reusable_fused_runtime_bundle(
    model_content_sha256=plan.identity().content().as_hex(),
    model_plan_sha256=plan.identity().plan().as_hex(),
    target=blueprint.target(),
    modules~,
  )
  @luna_fused_artifact_admission.admit_digest_pinned_reusable_fused_runtime_bundle(
    canonical,
    @crypto.bytes_to_hex_string(@crypto.sha256(canonical)),
  )
}

///|
fn qwen_fused_test_binding(
  space : PagedKvOperandSpace,
  offset : Int64,
  bytes : Int64,
) -> PagedKvOperandBinding {
  {
    role: Workspace,
    space,
    offset_bytes: offset,
    byte_count: bytes,
    alignment: 16L,
  }
}

///|
test "reusable Qwen runtime expands one module pair over every identical layer" {
  let plan = qwen_fused_test_plan()
  let blueprint = qwen_fused_test_blueprint(plan)
  let runtime = qwen_fused_test_runtime(plan, blueprint)
  let spans = admit_reusable_qwen_fused_attention_spans(
    plan,
    @worker_protocol.ModelPlanGeneration::first(),
    blueprint,
    runtime,
  )
  assert_eq(spans.length(), 28)
  assert_eq(spans[0].operation_indices(), (2, 3, 4, 5))
  assert_eq(spans[27].operation_indices(), (272, 273, 274, 275))
  for span in spans {
    assert_true(span.same_runtime_modules(spans[0]))
  }
  let split_decode = FixedArray::make(blueprint.steps().length(), -1)
  split_decode[5] = 99
  let mut prepared_step_count = 2
  for
    operation in [
      blueprint.steps()[2].operation_id(),
      blueprint.steps()[5].operation_id(),
    ] {
    if paged_production_split_decode_function(
        operation, blueprint, split_decode, spans,
      ) >=
      0 {
      prepared_step_count += 1
    }
  }
  assert_eq(prepared_step_count, 2)
  assert_true(qwen_fused_attention_overlaps_residual(spans, 3, 4))
  assert_false(qwen_fused_attention_overlaps_residual(spans, 7, 8))
}

///|
test "reusable Qwen runtime requires both ingress and readonly modules" {
  let plan = qwen_fused_test_plan()
  let blueprint = qwen_fused_test_blueprint(plan)
  let runtime = qwen_fused_test_runtime(plan, blueprint, include_readonly=false)
  try
    ignore(
      admit_reusable_qwen_fused_attention_spans(
        plan,
        @worker_protocol.ModelPlanGeneration::first(),
        blueprint,
        runtime,
      ),
    )
  catch {
    MismatchedFusedGroupedOperation(KernelAbi) => ()
    _ => fail("unexpected missing Qwen readonly rejection")
  } noraise {
    _ => fail("Qwen ingress-only bundle reached executable span admission")
  }
}

///|
test "reusable Qwen runtime requires head-parallel launch geometry" {
  let plan = qwen_fused_test_plan()
  let blueprint = qwen_fused_test_blueprint(plan)
  let runtime = qwen_fused_test_runtime(plan, blueprint, ingress_grid_y=1)
  try
    ignore(
      admit_reusable_qwen_fused_attention_spans(
        plan,
        @worker_protocol.ModelPlanGeneration::first(),
        blueprint,
        runtime,
      ),
    )
  catch {
    MismatchedFusedGroupedOperation(Launch) => ()
    _ => fail("unexpected Qwen launch rejection")
  } noraise {
    _ => fail("non-head-parallel Qwen launch reached executable admission")
  }
}

///|
test "shared runtime bundle preserves reusable residual fusion" {
  let plan = qwen_fused_test_plan()
  let blueprint = qwen_fused_test_blueprint(plan)
  let runtime = qwen_fused_test_runtime(plan, blueprint, include_residual=true)
  let spans = admit_reusable_fused_residual_rmsnorm_bundle_spans(
    plan,
    @worker_protocol.ModelPlanGeneration::first(),
    blueprint,
    runtime,
  )
  assert_true(!spans.is_empty())
  assert_eq(spans[0].operation_indices(), (7, 8))
}

///|
test "Qwen postprocess replacement preserves fast projection and rejects overlap" {
  let baseline = ReadOnlyArray::from_array([
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
  ])
  let replaced = replace_qwen_fused_attention_spans(
    baseline,
    ReadOnlyArray::from_array([2, 8]),
    ReadOnlyArray::from_array([20, 80]),
    ReadOnlyArray::from_array([50, 110]),
  )
  assert_eq(replaced, [0, 1, 2, 20, 50, 6, 7, 8, 80, 110])
  try
    ignore(
      replace_qwen_fused_attention_spans(
        baseline,
        ReadOnlyArray::from_array([2, 4]),
        ReadOnlyArray::from_array([20, 40]),
        ReadOnlyArray::from_array([50, 70]),
      ),
    )
  catch {
    InvalidExecutor(ResourceState) => ()
    _ => fail("unexpected overlapping Qwen replacement rejection")
  } noraise {
    _ => fail("overlapping Qwen replacements reached ordered execution")
  }
}

///|
test "Qwen fused sidecar preserves the complete rotated QKV region" {
  let rotated = qwen_fused_test_binding(ActivationWorkspaceArena, 64L, 2048L)
  let readonly_binding = qwen_fused_test_binding(
    ActivationWorkspaceArena,
    64L,
    2048L,
  )
  assert_eq(
    qwen_fused_attention_sidecar_shape_from_bindings(rotated, readonly_binding),
    (2048L, 16L),
  )
}
