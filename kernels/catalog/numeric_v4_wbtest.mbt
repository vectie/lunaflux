///|
fn numeric_v4_predicate_contract(
  activation : @numeric_contract.ComputeDType,
  accumulator : @numeric_contract.AccumulatorDType,
  output : @numeric_contract.ComputeDType,
) -> @numeric_contract.OperationExecutionContract {
  try! @numeric_contract.OperationExecutionContract::new(
    activation_input=activation,
    activation_compute=activation,
    activation_scale=@numeric_contract.ActivationScalePolicy::absent_exact_v1(),
    tensor_input=@numeric_contract.ComputeDType::bfloat16(),
    accumulator~,
    output~,
    conversion=@numeric_contract.NumericConversion::exact_v1(),
    order=@numeric_contract.OperationOrderPolicy::strict_declared_order_v1(),
  )
}

///|
test "legacy predicate detects every constructible non-tensor integer position" {
  let bf16 = @numeric_contract.ComputeDType::bfloat16()
  let i8 = @numeric_contract.ComputeDType::i8()
  let f32_accumulator = @numeric_contract.AccumulatorDType::f32()
  let hostile = [
    numeric_v4_predicate_contract(i8, f32_accumulator, bf16),
    numeric_v4_predicate_contract(
      bf16,
      @numeric_contract.AccumulatorDType::i32(),
      bf16,
    ),
    numeric_v4_predicate_contract(bf16, f32_accumulator, i8),
  ]
  for execution in hostile {
    assert_true(execution_contract_uses_i8_or_integer_accumulator(execution))
  }
}

///|
test "legacy predicate keeps BF16 and FP8 outside the I8 boundary" {
  let bf16 = numeric_v4_predicate_contract(
    @numeric_contract.ComputeDType::bfloat16(),
    @numeric_contract.AccumulatorDType::f32(),
    @numeric_contract.ComputeDType::bfloat16(),
  )
  assert_false(execution_contract_uses_i8_or_integer_accumulator(bf16))
  let fp8 = try! @numeric_contract.OperationExecutionContract::new(
    activation_input=@numeric_contract.ComputeDType::bfloat16(),
    activation_compute=@numeric_contract.ComputeDType::f8_e4m3_finite(),
    activation_scale=@numeric_contract.ActivationScalePolicy::dynamic_per_tensor_f32_v1(),
    tensor_input=@numeric_contract.ComputeDType::f8_e4m3_finite(),
    accumulator=@numeric_contract.AccumulatorDType::f32(),
    output=@numeric_contract.ComputeDType::bfloat16(),
    conversion=@numeric_contract.NumericConversion::quantized_v1(),
    order=@numeric_contract.OperationOrderPolicy::strict_declared_order_v1(),
  )
  assert_false(execution_contract_uses_i8_or_integer_accumulator(fp8))
}
