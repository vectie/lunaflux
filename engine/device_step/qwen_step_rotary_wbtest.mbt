///|
test "step rotary cache uses one bounded append shared by all compatible layers" {
  let plan = qwen_fused_test_plan()
  let blueprint = qwen_fused_test_blueprint(plan)
  let legacy = admit_reusable_qwen_fused_attention_spans(
    plan,
    @worker_protocol.ModelPlanGeneration::first(),
    blueprint,
    qwen_fused_test_runtime(plan, blueprint, full_ingress=true),
  )
  let cached = admit_reusable_qwen_fused_attention_spans(
    plan,
    @worker_protocol.ModelPlanGeneration::first(),
    blueprint,
    qwen_fused_test_runtime(
      plan,
      blueprint,
      full_ingress=true,
      step_rotary_cache=true,
    ),
  )
  assert_eq(cached.length(), legacy.length())
  let cache = cached[0].step_rotary_cache.unwrap()
  assert_eq(cache.geometry(), (blueprint.profile().max_query_tokens(), 64))
  assert_eq(
    cache.byte_count(),
    blueprint.profile().max_query_tokens().to_int64() * 512L,
  )
  let rotated_binding = qwen_fused_test_binding(
    ActivationWorkspaceArena,
    64L,
    2048L,
  )
  let readonly_binding = qwen_fused_test_binding(
    ActivationWorkspaceArena,
    64L,
    2048L,
  )
  let rotated = qwen_fused_attention_sidecar_shape_from_bindings(
    rotated_binding, readonly_binding,
  )
  let complete = qwen_step_rotary_sidecar_shape(rotated, Some(cache))
  assert_eq(complete.0 - cache.byte_count(), rotated.0)
  assert_eq(complete.0, rotated.0 + cache.byte_count())
  assert_eq(qwen_step_rotary_sidecar_shape(rotated, None), rotated)
  for span in cached {
    assert_true(span.same_runtime_modules(cached[0]))
  }
  let launch = qwen_step_rotary_launch(cache)
  assert_eq(launch.block_x(), 128)
  assert_eq(launch.grid_x(), (blueprint.profile().max_query_tokens() + 1) / 2)
  let rule = QueryCellCappedGridX(launch, 64)
  assert_eq(paged_projection_bucket_grid_x(rule, 1), 1)
  assert_eq(paged_projection_bucket_grid_x(rule, 3), 2)
  let preparation_operation = qwen_step_rotary_operation(blueprint)
  assert_false(
    blueprint.steps().any(step => step.operation_id() == preparation_operation),
  )
  assert_eq(
    qwen_step_rotary_bytes_for_operation(
      blueprint,
      blueprint.steps()[5].operation_id(),
      cached,
    ),
    cache.byte_count(),
  )
}

///|
test "cell-based preparation buckets preserve non-divisor head widths" {
  let cache = @elementwise_physical_ir.StepRotaryCachePlan::new(
    maximum_tokens=32,
    head_dimension=96,
  ).unwrap()
  let launch = qwen_step_rotary_launch(cache)
  let rule = QueryCellCappedGridX(launch, 48)
  assert_eq(paged_projection_bucket_grid_x(rule, 1), 1)
  assert_eq(paged_projection_bucket_grid_x(rule, 2), 1)
  assert_eq(paged_projection_bucket_grid_x(rule, 3), 2)
  assert_eq(paged_projection_bucket_grid_x(rule, 32), 12)
}

///|
test "a single preparation precedes every borrowed layer without changing legacy routes" {
  let layers : ReadOnlyArray[Int] = [10, 20, 30, 40]
  assert_eq(
    prepend_qwen_step_rotary(layers, Some(9)),
    ReadOnlyArray::from_array([9, 10, 20, 30, 40]),
  )
  assert_eq(prepend_qwen_step_rotary(layers, None), layers)
  let prefill : ReadOnlyArray[Int] = [10, 21, 30, 41]
  let decode : ReadOnlyArray[Int] = [10, 22, 30, 42]
  for route in [layers, prefill, decode] {
    let prepared = prepend_qwen_step_rotary(route, Some(9))
    assert_eq(prepared[0], 9)
    assert_eq(prepared.filter(step => step == 9).length(), 1)
    assert_eq(prepared.length(), route.length() + 1)
  }
}
