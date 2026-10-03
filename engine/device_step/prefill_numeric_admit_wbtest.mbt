///|
test "approximate exponential does not grant dtype or operation order changes" {
  let plan = qwen_fused_test_plan()
  let attention = plan.numeric_binding().operations()[5].execution()
  assert_true(prefill_exponential_execution_compatible(attention))
  let different_output = @numeric_contract.OperationExecutionContract::new(
    activation_input=@numeric_contract.ComputeDType::bfloat16(),
    activation_compute=@numeric_contract.ComputeDType::bfloat16(),
    activation_scale=@numeric_contract.ActivationScalePolicy::absent_exact_v1(),
    tensor_input=@numeric_contract.ComputeDType::absent(),
    accumulator=@numeric_contract.AccumulatorDType::f32(),
    output=@numeric_contract.ComputeDType::f32(),
    conversion=@numeric_contract.NumericConversion::exact_v1(),
    order=@numeric_contract.OperationOrderPolicy::strict_declared_order_v1(),
  )
  assert_false(prefill_exponential_execution_compatible(different_output))
}

///|
test "approximate prefill startup checks the immutable attention numeric binding" {
  let plan = qwen_fused_test_plan()
  let blueprint = qwen_fused_test_blueprint(plan)
  let module_ = @luna_fused_artifact_admission.ReusableFusedRuntimeModuleInput::new(
    abi=QwenPagedAttentionReadOnlyProductionV2,
    first_operation=5,
    operation_count=1,
    function_symbol="lunaflux_attention_prefill_tile_compiler_exp2_v1",
    query_tile_rows=16,
    launch=@launch_contract.AotLaunchDimensions::new(
      grid_x=32,
      grid_y=16,
      grid_z=1,
      block_x=128,
      block_y=1,
      block_z=1,
      shared_memory_bytes=49168,
    ),
    module_bytes=b"startup-approximate-prefill-test",
    prefill_exponential_law=ApproximateBaseTwoExponential,
    allow_approximate_prefill_exponential=true,
  )
  let bytes = @luna_fused_artifact_admission.export_reusable_fused_runtime_bundle(
    model_content_sha256=plan.identity().content().as_hex(),
    model_plan_sha256=plan.identity().plan().as_hex(),
    target=blueprint.target(),
    modules=[module_],
  )
  let runtime = @luna_fused_artifact_admission.admit_digest_pinned_reusable_fused_runtime_bundle(
    bytes,
    @crypto.bytes_to_hex_string(@crypto.sha256(bytes)),
  )
  admit_prefill_exponential_model_contract(plan, runtime)
}
