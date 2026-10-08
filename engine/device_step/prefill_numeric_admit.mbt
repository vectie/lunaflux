///|
/// Both named prefill alternatives require BF16 external operands and an F32
/// accumulator. The reference law additionally names internal BF16 probability
/// materialization; it is not the elementary exp2 law. This external-contract
/// check grants neither a new operand dtype nor whole-model accuracy approval.
fn prefill_exponential_execution_compatible(
  execution : @numeric_contract.OperationExecutionContract,
) -> Bool {
  execution.activation_input() == @numeric_contract.ComputeDType::bfloat16() &&
  execution.activation_compute() == @numeric_contract.ComputeDType::bfloat16() &&
  execution.activation_scale() ==
  @numeric_contract.ActivationScalePolicy::absent_exact_v1() &&
  execution.accumulator() == @numeric_contract.AccumulatorDType::f32() &&
  execution.output() == @numeric_contract.ComputeDType::bfloat16() &&
  execution.conversion() == @numeric_contract.NumericConversion::exact_v1() &&
  execution.order() ==
  @numeric_contract.OperationOrderPolicy::strict_declared_order_v1()
}

///|
/// AOT numerical permission is consumed once, before creating reusable spans.
/// It cannot silently broaden the immutable model operation contract.
fn admit_prefill_exponential_model_contract(
  plan : @plan.ModelPlan,
  runtime : @luna_fused_artifact_admission.ApprovedReusableFusedRuntimeBundle,
) -> Unit raise FusedGroupedOperationError {
  let approximate = runtime
    .modules()
    .any(module_ => {
      module_.prefill_exponential_law() != StrictNaturalExponential
    })
  if !approximate {
    return
  }
  let mut attention_count = 0
  let operations = plan.operations()
  for binding in plan.numeric_binding().operations() {
    let ordinal = binding.operation().as_int()
    guard ordinal >= 0 && ordinal < operations.length() else {
      raise MismatchedFusedGroupedOperation(KernelAbi)
    }
    if operations[ordinal].kind() == CausalAttention {
      attention_count = attention_count + 1
      if !prefill_exponential_execution_compatible(binding.execution()) {
        raise MismatchedFusedGroupedOperation(KernelAbi)
      }
    }
  }
  if attention_count == 0 {
    raise MismatchedFusedGroupedOperation(KernelAbi)
  }
}
