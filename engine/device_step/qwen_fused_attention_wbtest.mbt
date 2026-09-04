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
  phase_specific_attention? : Bool = false,
  direct_compiler_decode? : Bool = false,
  split_compiler_decode? : Bool = false,
  full_ingress? : Bool = false,
  wide_query_prefill? : Bool = false,
  wide_query_prefill_alias_selected? : Bool = false,
  partitioned_prefill? : Bool = false,
  partitioned_prefill_grid_x? : Int = -1,
  deep_partitioned_prefill? : Bool = false,
) -> @luna_fused_artifact_admission.ApprovedReusableFusedRuntimeBundle raise {
  let ingress = @luna_fused_artifact_admission.ReusableFusedRuntimeModuleInput::new(
    abi=if full_ingress {
      QwenQkvQkNormPositionedRopePagedKvWriteProductionV4
    } else {
      QwenQkNormPositionedRopePagedKvWriteProductionV2
    },
    first_operation=if full_ingress { 2 } else { 3 },
    operation_count=if full_ingress { 4 } else { 3 },
    function_symbol=if full_ingress {
      "qwen_full_ingress"
    } else {
      "qwen_postprocess"
    },
    launch=@launch_contract.AotLaunchDimensions::new(
      grid_x=if full_ingress {
        (blueprint.profile().max_query_tokens() + 15) / 16
      } else {
        blueprint.profile().max_query_tokens()
      },
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
        function_symbol=if phase_specific_attention {
          "qwen_compiler_prefill"
        } else {
          "qwen_readonly"
        },
        launch=@launch_contract.AotLaunchDimensions::new(
          grid_x=if phase_specific_attention {
            blueprint.profile().max_query_rows() +
            (
              blueprint.profile().max_query_tokens() -
              blueprint.profile().max_query_rows() +
              15
            ) /
            16
          } else {
            Int::min(blueprint.profile().max_query_tokens(), 128)
          },
          grid_y=16,
          grid_z=1,
          block_x=if phase_specific_attention { 256 } else { 128 },
          block_y=1,
          block_z=1,
          shared_memory_bytes=if phase_specific_attention { 31952 } else { 0 },
        ),
        module_bytes=b"qwen-readonly-module",
      ),
    )
  }
  if phase_specific_attention {
    modules.push(
      @luna_fused_artifact_admission.ReusableFusedRuntimeModuleInput::new(
        abi=QwenPagedAttentionReadOnlyDecodeSplitProductionV3,
        first_operation=5,
        operation_count=1,
        function_symbol="lunaflux_attention_decode_tile_compiler_v1",
        launch=@launch_contract.AotLaunchDimensions::new(
          grid_x=blueprint.profile().max_query_rows(),
          grid_y=if direct_compiler_decode { 16 } else { 8 },
          grid_z=1,
          block_x=if direct_compiler_decode {
            32
          } else if split_compiler_decode {
            256
          } else {
            64
          },
          block_y=1,
          block_z=1,
          shared_memory_bytes=if direct_compiler_decode { 0 } else { 34076 },
        ),
        module_bytes=b"qwen-decode-module",
      ),
    )
  }
  if wide_query_prefill {
    modules.push(
      @luna_fused_artifact_admission.ReusableFusedRuntimeModuleInput::new(
        abi=QwenPagedAttentionPrefillWideQueryProductionV5,
        first_operation=5,
        operation_count=1,
        function_symbol=if wide_query_prefill_alias_selected {
          "qwen_compiler_prefill"
        } else {
          "lunaflux_attention_prefill_tile_compiler_v1_c314"
        },
        launch=@launch_contract.AotLaunchDimensions::new(
          grid_x=blueprint.profile().max_query_rows() +
            (
              blueprint.profile().max_query_tokens() -
              blueprint.profile().max_query_rows() +
              15
            ) /
            16,
          grid_y=16,
          grid_z=1,
          block_x=256,
          block_y=1,
          block_z=1,
          shared_memory_bytes=if wide_query_prefill_alias_selected {
            31952
          } else {
            70032
          },
        ),
        module_bytes=if wide_query_prefill_alias_selected {
          b"qwen-readonly-module"
        } else {
          b"qwen-wide-prefill-module"
        },
      ),
    )
  }
  if partitioned_prefill {
    let shared_module = b"qwen-partitioned-prefill-module"
    let maximum_prefill_grid = blueprint.profile().max_query_rows() +
      (
        blueprint.profile().max_query_tokens() -
        blueprint.profile().max_query_rows() +
        15
      ) /
      16
    let prefill_grid = if partitioned_prefill_grid_x > 0 {
      partitioned_prefill_grid_x
    } else {
      maximum_prefill_grid
    }
    modules.push(
      @luna_fused_artifact_admission.ReusableFusedRuntimeModuleInput::new(
        abi=QwenPagedAttentionPrefillPartitionedPartialProductionV4,
        first_operation=5,
        operation_count=1,
        function_symbol="lunaflux_attention_prefill_partitioned_partial_v1",
        launch=@launch_contract.AotLaunchDimensions::new(
          grid_x=prefill_grid,
          grid_y=16,
          grid_z=2,
          block_x=256,
          block_y=1,
          block_z=1,
          shared_memory_bytes=31952,
        ),
        module_bytes=shared_module,
      ),
    )
    modules.push(
      @luna_fused_artifact_admission.ReusableFusedRuntimeModuleInput::new(
        abi=QwenPagedAttentionPrefillPartitionedMergeProductionV4,
        first_operation=5,
        operation_count=1,
        function_symbol="lunaflux_attention_prefill_partitioned_merge_v1",
        launch=@launch_contract.AotLaunchDimensions::new(
          grid_x=prefill_grid,
          grid_y=16,
          grid_z=1,
          block_x=256,
          block_y=1,
          block_z=1,
          shared_memory_bytes=0,
        ),
        module_bytes=shared_module,
      ),
    )
  }
  if deep_partitioned_prefill {
    let shared_module = b"qwen-deep-partitioned-prefill-module"
    let prefill_grid = if partitioned_prefill_grid_x > 0 {
      partitioned_prefill_grid_x
    } else {
      blueprint.profile().max_query_rows() +
      (
        blueprint.profile().max_query_tokens() -
        blueprint.profile().max_query_rows() +
        15
      ) /
      16
    }
    modules.push(
      @luna_fused_artifact_admission.ReusableFusedRuntimeModuleInput::new(
        abi=QwenPagedAttentionPrefillPartitioned8PartialProductionV6,
        first_operation=5,
        operation_count=1,
        function_symbol="lunaflux_attention_prefill_partitioned_partial_v1_p8",
        launch=@launch_contract.AotLaunchDimensions::new(
          grid_x=prefill_grid,
          grid_y=16,
          grid_z=8,
          block_x=256,
          block_y=1,
          block_z=1,
          shared_memory_bytes=31952,
        ),
        module_bytes=shared_module,
      ),
    )
    modules.push(
      @luna_fused_artifact_admission.ReusableFusedRuntimeModuleInput::new(
        abi=QwenPagedAttentionPrefillPartitioned8MergeProductionV6,
        first_operation=5,
        operation_count=1,
        function_symbol="lunaflux_attention_prefill_partitioned_merge_v1_p8",
        launch=@launch_contract.AotLaunchDimensions::new(
          grid_x=prefill_grid,
          grid_y=16,
          grid_z=1,
          block_x=256,
          block_y=1,
          block_z=1,
          shared_memory_bytes=0,
        ),
        module_bytes=shared_module,
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
test "reusable Qwen runtime admits split grouped compiler decode kernel" {
  let plan = qwen_fused_test_plan()
  let blueprint = qwen_fused_test_blueprint(plan)
  let runtime = qwen_fused_test_runtime(
    plan,
    blueprint,
    phase_specific_attention=true,
    split_compiler_decode=true,
  )
  let spans = admit_reusable_qwen_fused_attention_spans(
    plan,
    @worker_protocol.ModelPlanGeneration::first(),
    blueprint,
    runtime,
  )
  assert_eq(spans.length(), 28)
  assert_eq(spans[0].decode_launch().grid_y(), 8)
  assert_eq(spans[0].decode_launch().block_x(), 256)
  assert_eq(spans[0].decode_launch().shared_memory_bytes(), 34076)
}

///|
test "reusable Qwen runtime admits direct compiler decode kernel" {
  let plan = qwen_fused_test_plan()
  let blueprint = qwen_fused_test_blueprint(plan)
  let runtime = qwen_fused_test_runtime(
    plan,
    blueprint,
    phase_specific_attention=true,
    direct_compiler_decode=true,
  )
  let spans = admit_reusable_qwen_fused_attention_spans(
    plan,
    @worker_protocol.ModelPlanGeneration::first(),
    blueprint,
    runtime,
  )
  assert_eq(spans.length(), 28)
  assert_eq(spans[0].decode_launch().grid_y(), 16)
  assert_eq(spans[0].decode_launch().block_x(), 32)
  assert_eq(spans[0].decode_launch().shared_memory_bytes(), 0)
}

///|
test "reusable Qwen runtime admits wide-query prefill frontier member" {
  let plan = qwen_fused_test_plan()
  let blueprint = qwen_fused_test_blueprint(plan)
  let runtime = qwen_fused_test_runtime(
    plan,
    blueprint,
    phase_specific_attention=true,
    wide_query_prefill=true,
  )
  let spans = admit_reusable_qwen_fused_attention_spans(
    plan,
    @worker_protocol.ModelPlanGeneration::first(),
    blueprint,
    runtime,
  )
  guard spans[0].wide_query_prefill() is Some(variant) else {
    fail("wide-query prefill variant missing")
  }
  assert_eq(variant.query_tile_rows, 64)
  assert_eq(variant.launch.shared_memory_bytes(), 70032)
  assert_eq(qwen_wide_prefill_minimum_query_tokens(spans), 64)
}

///|
test "selected prefill alias does not create a fake wide-query executor" {
  let plan = qwen_fused_test_plan()
  let blueprint = qwen_fused_test_blueprint(plan)
  let runtime = qwen_fused_test_runtime(
    plan,
    blueprint,
    phase_specific_attention=true,
    full_ingress=true,
    wide_query_prefill=true,
    wide_query_prefill_alias_selected=true,
  )
  let spans = admit_reusable_qwen_fused_attention_spans(
    plan,
    @worker_protocol.ModelPlanGeneration::first(),
    blueprint,
    runtime,
  )
  assert_true(spans[0].wide_query_prefill() is None)
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
  let first_attention = blueprint.steps()[5]
  assert_true(
    paged_decode_split_uses_qwen_sidecar(
      blueprint,
      first_attention.operation_id(),
      spans[0],
    ),
  )
  assert_false(
    paged_decode_split_uses_qwen_sidecar(
      blueprint,
      blueprint.steps()[6].operation_id(),
      spans[0],
    ),
  )
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
test "compiler-fused Qwen ingress replaces projection through KV write" {
  let plan = qwen_fused_test_plan()
  let blueprint = qwen_fused_test_blueprint(plan)
  let runtime = qwen_fused_test_runtime(plan, blueprint, full_ingress=true)
  let spans = admit_reusable_qwen_fused_attention_spans(
    plan,
    @worker_protocol.ModelPlanGeneration::first(),
    blueprint,
    runtime,
  )
  assert_eq(spans.length(), 28)
  assert_true(spans[0].ingress_includes_projection())
  assert_eq(spans[0].operation_indices(), (2, 3, 4, 5))
  assert_eq(spans[0].postprocess_function_symbol(), "qwen_full_ingress")
  let baseline : Array[Int] = []
  for index in 0..<blueprint.steps().length() {
    baseline.push(index)
  }
  let ingress = Array::make(spans.length(), 10000)
  let attention_steps = Array::make(spans.length(), 20000)
  let replaced = replace_qwen_fused_attention_runtime_spans(
    ReadOnlyArray::from_array(baseline),
    spans,
    ReadOnlyArray::from_array(ingress),
    ReadOnlyArray::from_array(attention_steps),
  )
  assert_eq(replaced.length(), blueprint.steps().length() - 2 * spans.length())
  assert_eq(replaced[0], 0)
  assert_eq(replaced[1], 1)
  assert_eq(replaced[2], 10000)
  assert_eq(replaced[3], 20000)
  assert_eq(replaced[4], 6)
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
test "reusable Qwen runtime admits separate compiler prefill and decode kernels" {
  let plan = qwen_fused_test_plan()
  let blueprint = qwen_fused_test_blueprint(plan)
  let runtime = qwen_fused_test_runtime(
    plan,
    blueprint,
    phase_specific_attention=true,
  )
  let spans = admit_reusable_qwen_fused_attention_spans(
    plan,
    @worker_protocol.ModelPlanGeneration::first(),
    blueprint,
    runtime,
  )
  assert_eq(spans.length(), 28)
  assert_true(spans[0].has_phase_specific_attention())
  assert_eq(spans[0].readonly_function_symbol(), "qwen_compiler_prefill")
  assert_eq(
    spans[0].decode_function_symbol(),
    "lunaflux_attention_decode_tile_compiler_v1",
  )
  assert_eq(spans[0].readonly_launch().block_x(), 256)
  assert_eq(spans[0].decode_launch().block_x(), 64)
  assert_eq(
    spans[0].decode_launch().grid_x(),
    blueprint.profile().max_query_rows(),
  )
  assert_eq(spans[0].decode_launch().shared_memory_bytes(), 34076)
}

///|
test "Qwen runtime admits compiler partition-map fold merge prefill" {
  let plan = qwen_fused_test_plan()
  let blueprint = qwen_fused_test_blueprint(plan)
  let runtime = qwen_fused_test_runtime(
    plan,
    blueprint,
    phase_specific_attention=true,
    full_ingress=true,
    partitioned_prefill=true,
  )
  let spans = admit_reusable_qwen_fused_attention_spans(
    plan,
    @worker_protocol.ModelPlanGeneration::first(),
    blueprint,
    runtime,
  )
  guard spans[0].partitioned_prefill() is Some(pipeline) else {
    fail("partitioned Qwen prefill missing")
  }
  assert_eq(pipeline.partial_launch.grid_z(), 2)
  assert_eq(pipeline.merge_launch.grid_z(), 1)
  assert_eq(pipeline.merge_launch.shared_memory_bytes(), 0)
  assert_true(pipeline.workspace_bytes > 0L)
  assert_eq(
    qwen_partitioned_prefill_minimum_context_tokens(blueprint, spans),
    2 * 16 * blueprint.layout().tokens_per_page(),
  )
  assert_eq(qwen_partitioned_prefill_query_tile(blueprint, spans), 16)
  assert_eq(qwen_partitioned_prefill_partition_count(spans), 2)
}

///|
test "Qwen runtime admits an ordered deep partition frontier member" {
  let plan = qwen_fused_test_plan()
  let blueprint = qwen_fused_test_blueprint(plan)
  let runtime = qwen_fused_test_runtime(
    plan,
    blueprint,
    phase_specific_attention=true,
    full_ingress=true,
    partitioned_prefill=true,
    deep_partitioned_prefill=true,
  )
  let spans = admit_reusable_qwen_fused_attention_spans(
    plan,
    @worker_protocol.ModelPlanGeneration::first(),
    blueprint,
    runtime,
  )
  guard spans[0].partitioned_prefill() is Some(base) &&
    spans[0].deep_partitioned_prefill() is Some(deep) else {
    fail("partitioned Qwen frontier missing")
  }
  assert_eq(base.partial_launch.grid_z(), 2)
  assert_eq(deep.partial_launch.grid_z(), 8)
  assert_true(deep.workspace_bytes > base.workspace_bytes)
  assert_eq(qwen_partitioned_prefill_partition_count(spans, deep=true), 8)
}

///|
test "Qwen runtime admits low-parallelism partition grid specialization" {
  let plan = qwen_fused_test_plan()
  let blueprint = qwen_fused_test_blueprint(plan)
  let runtime = qwen_fused_test_runtime(
    plan,
    blueprint,
    phase_specific_attention=true,
    full_ingress=true,
    partitioned_prefill=true,
    partitioned_prefill_grid_x=2,
  )
  let spans = admit_reusable_qwen_fused_attention_spans(
    plan,
    @worker_protocol.ModelPlanGeneration::first(),
    blueprint,
    runtime,
  )
  guard spans[0].partitioned_prefill() is Some(pipeline) else {
    fail("specialized partitioned Qwen prefill missing")
  }
  assert_eq(pipeline.partial_launch.grid_x(), 2)
  assert_eq(pipeline.merge_launch.grid_x(), 2)
  assert_eq(qwen_partitioned_prefill_query_tile(blueprint, spans), 16)
  assert_eq(qwen_partitioned_prefill_partition_count(spans), 2)
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
  let replaced = replace_projection_postprocess_attention_spans(
    baseline,
    ReadOnlyArray::from_array([2, 8]),
    ReadOnlyArray::from_array([20, 80]),
    ReadOnlyArray::from_array([50, 110]),
  )
  assert_eq(replaced, [0, 1, 2, 20, 50, 6, 7, 8, 80, 110])
  try
    ignore(
      replace_projection_postprocess_attention_spans(
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
