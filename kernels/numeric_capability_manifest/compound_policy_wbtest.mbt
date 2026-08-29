///|
fn manifest_policy_execution(
  policy : @numeric_contract.ActivationScalePolicy,
) -> @numeric_contract.OperationExecutionContract {
  try! @numeric_contract.OperationExecutionContract::new(
    activation_input=@numeric_contract.ComputeDType::bfloat16(),
    activation_compute=@numeric_contract.ComputeDType::f8_e4m3_finite(),
    activation_scale=policy,
    tensor_input=@numeric_contract.ComputeDType::f8_e4m3_finite(),
    accumulator=@numeric_contract.AccumulatorDType::f32(),
    output=@numeric_contract.ComputeDType::bfloat16(),
    conversion=@numeric_contract.NumericConversion::quantized_v1(),
    order=@numeric_contract.OperationOrderPolicy::strict_declared_order_v1(),
  )
}

///|
fn manifest_policy_operation(
  kind : @plan.OperationKind,
  workspace : Int64,
) -> @plan.PlanOperation {
  let id = try! @plan.OperationId::new(0)
  match kind {
    GatedMlp =>
      try! @plan.PlanOperation::new_stateless(
        id~,
        kind~,
        inputs=[
          Activation(@plan.ActivationRef::new(0)),
          Tensor(@plan.TensorRef::new(0)),
          Tensor(@plan.TensorRef::new(1)),
          Tensor(@plan.TensorRef::new(2)),
        ],
        outputs=[@plan.ActivationRef::new(1)],
        shape=GatedMlpShape(hidden_size=16, intermediate_size=32),
        required_capability=@plan.KernelCapabilityId::gated_mlp(),
        workspace_bytes=workspace,
      )
    OutputProjection =>
      try! @plan.PlanOperation::new_stateless(
        id~,
        kind~,
        inputs=[
          Activation(@plan.ActivationRef::new(0)),
          Tensor(@plan.TensorRef::new(0)),
        ],
        outputs=[@plan.ActivationRef::new(1)],
        shape=ProjectionShape(input_width=16, output_width=16),
        required_capability=@plan.KernelCapabilityId::dense_projection(),
        workspace_bytes=workspace,
      )
    _ => abort("unsupported manifest policy fixture kind")
  }
}

///|
test "manifest retains legacy simple GatedMlp and closes compound kind pairing" {
  let simple = manifest_policy_execution(
    @numeric_contract.ActivationScalePolicy::dynamic_per_tensor_f32_v1(),
  )
  let compound = manifest_policy_execution(
    @numeric_contract.ActivationScalePolicy::gated_mlp_external_and_post_silu_f32_v1(),
  )
  assert_true(
    exact_fp8_w8a8_execution(manifest_policy_operation(GatedMlp, 0L), simple),
  )
  assert_true(
    exact_fp8_w8a8_execution(manifest_policy_operation(GatedMlp, 8L), compound),
  )
  assert_false(
    exact_fp8_w8a8_execution(manifest_policy_operation(GatedMlp, 4L), compound),
  )
  assert_false(
    exact_fp8_w8a8_execution(
      manifest_policy_operation(OutputProjection, 8L),
      compound,
    ),
  )
  assert_true(mentions_fp8_execution(compound))
}
