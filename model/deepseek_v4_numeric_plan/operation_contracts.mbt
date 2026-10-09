///|
fn exact_execution(
  activation : @numeric_contract.ComputeDType,
  tensor : @numeric_contract.ComputeDType,
  output : @numeric_contract.ComputeDType,
) -> @numeric_contract.OperationExecutionContract {
  try! @numeric_contract.OperationExecutionContract::new(
    activation_input=activation,
    activation_compute=activation,
    activation_scale=@numeric_contract.ActivationScalePolicy::absent_exact_v1(),
    tensor_input=tensor,
    accumulator=@numeric_contract.AccumulatorDType::f32(),
    output~,
    conversion=@numeric_contract.NumericConversion::exact_v1(),
    order=@numeric_contract.OperationOrderPolicy::strict_declared_order_v1(),
  )
}

///|
fn bf16_tensor_execution() -> @numeric_contract.OperationExecutionContract {
  exact_execution(
    @numeric_contract.ComputeDType::bfloat16(),
    @numeric_contract.ComputeDType::bfloat16(),
    @numeric_contract.ComputeDType::bfloat16(),
  )
}

///|
fn bf16_activation_execution() -> @numeric_contract.OperationExecutionContract {
  exact_execution(
    @numeric_contract.ComputeDType::bfloat16(),
    @numeric_contract.ComputeDType::absent(),
    @numeric_contract.ComputeDType::bfloat16(),
  )
}

///|
fn bf16_parameter_execution() -> @numeric_contract.OperationExecutionContract {
  exact_execution(
    @numeric_contract.ComputeDType::absent(),
    @numeric_contract.ComputeDType::bfloat16(),
    @numeric_contract.ComputeDType::bfloat16(),
  )
}

///|
fn bf16_f32_control_execution() -> @numeric_contract.OperationExecutionContract {
  exact_execution(
    @numeric_contract.ComputeDType::bfloat16(),
    @numeric_contract.ComputeDType::f32(),
    @numeric_contract.ComputeDType::bfloat16(),
  )
}

///|
fn bf16_f32_control_to_f32_execution() -> @numeric_contract.OperationExecutionContract {
  exact_execution(
    @numeric_contract.ComputeDType::bfloat16(),
    @numeric_contract.ComputeDType::f32(),
    @numeric_contract.ComputeDType::f32(),
  )
}

///|
fn bf16_router_execution() -> @numeric_contract.OperationExecutionContract {
  exact_execution(
    @numeric_contract.ComputeDType::bfloat16(),
    @numeric_contract.ComputeDType::bfloat16(),
    @numeric_contract.ComputeDType::f32(),
  )
}

///|
fn f32_activation_execution() -> @numeric_contract.OperationExecutionContract {
  exact_execution(
    @numeric_contract.ComputeDType::f32(),
    @numeric_contract.ComputeDType::absent(),
    @numeric_contract.ComputeDType::f32(),
  )
}

///|
fn fp8_projection_execution() -> @numeric_contract.OperationExecutionContract {
  try! @numeric_contract.OperationExecutionContract::new(
    activation_input=@numeric_contract.ComputeDType::bfloat16(),
    activation_compute=@numeric_contract.ComputeDType::f8_e4m3_finite(),
    activation_scale=@numeric_contract.ActivationScalePolicy::dynamic_per_tensor_f32_v1(),
    tensor_input=@numeric_contract.ComputeDType::f8_e4m3_finite(),
    accumulator=@numeric_contract.AccumulatorDType::f32(),
    output=@numeric_contract.ComputeDType::bfloat16(),
    conversion=@numeric_contract.NumericConversion::quantized_v1(),
    order=@numeric_contract.OperationOrderPolicy::strict_declared_order_v1(),
  )
}

///|
fn fp8_block128_ue8m0_projection_execution() -> @numeric_contract.OperationExecutionContract {
  try! @numeric_contract.OperationExecutionContract::new(
    activation_input=@numeric_contract.ComputeDType::bfloat16(),
    activation_compute=@numeric_contract.ComputeDType::f8_e4m3_finite(),
    activation_scale=@numeric_contract.ActivationScalePolicy::dynamic_per_row_block128_ue8m0_v1(),
    tensor_input=@numeric_contract.ComputeDType::f8_e4m3_finite(),
    accumulator=@numeric_contract.AccumulatorDType::f32(),
    output=@numeric_contract.ComputeDType::bfloat16(),
    conversion=@numeric_contract.NumericConversion::quantized_v1(),
    order=@numeric_contract.OperationOrderPolicy::strict_declared_order_v1(),
  )
}

///|
fn fp8_gated_execution() -> @numeric_contract.OperationExecutionContract {
  try! @numeric_contract.OperationExecutionContract::new(
    activation_input=@numeric_contract.ComputeDType::bfloat16(),
    activation_compute=@numeric_contract.ComputeDType::f8_e4m3_finite(),
    activation_scale=@numeric_contract.ActivationScalePolicy::gated_mlp_external_and_post_silu_bound_expf_f32_v2(),
    tensor_input=@numeric_contract.ComputeDType::f8_e4m3_finite(),
    accumulator=@numeric_contract.AccumulatorDType::f32(),
    output=@numeric_contract.ComputeDType::bfloat16(),
    conversion=@numeric_contract.NumericConversion::quantized_v1(),
    order=@numeric_contract.OperationOrderPolicy::strict_declared_order_v1(),
  )
}
