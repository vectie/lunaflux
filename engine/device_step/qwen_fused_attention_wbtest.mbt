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
  let baseline : Array[Int32] = []
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
  assert_eq(qwen_partitioned_prefill_query_tile(blueprint, spans), 32)
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
  assert_eq(qwen_partitioned_prefill_query_tile(blueprint, spans), 32)
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
