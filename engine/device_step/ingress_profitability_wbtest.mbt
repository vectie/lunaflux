///|
test "unfused ingress choice retains the entire baseline graph" {
  let plan = qwen_fused_test_plan()
  let blueprint = qwen_fused_test_blueprint(plan)
  let runtime = qwen_fused_test_runtime(
    plan,
    blueprint,
    include_residual=true,
    include_ingress=false,
    include_readonly=false,
  )
  let spans = admit_reusable_qwen_fused_attention_spans(
    plan,
    @worker_protocol.ModelPlanGeneration::first(),
    blueprint,
    runtime,
  )
  assert_true(spans.is_empty())
  let baseline = ReadOnlyArray::from_array([0, 1, 2, 3, 4, 5, 6, 7, 8])
  let replaced = replace_qwen_fused_attention_runtime_spans(
    baseline,
    spans,
    ReadOnlyArray::from_array([]),
    ReadOnlyArray::from_array([]),
  )
  assert_eq(replaced, baseline)
}

///|
test "unfused choice cannot strand readonly attention without KV writes" {
  let plan = qwen_fused_test_plan()
  let blueprint = qwen_fused_test_blueprint(plan)
  let runtime = qwen_fused_test_runtime(
    plan,
    blueprint,
    include_residual=true,
    include_ingress=false,
    include_readonly=true,
  )
  try
    admit_reusable_qwen_fused_attention_spans(
      plan,
      @worker_protocol.ModelPlanGeneration::first(),
      blueprint,
      runtime,
    )
  catch {
    error => assert_eq(error, MismatchedFusedGroupedOperation(KernelAbi))
  } noraise {
    _ => fail("readonly attention lost its KV write")
  }
}
