///|
fn test_storage(
  dtype : @numeric_contract.StorageDType,
  encoding : @numeric_contract.StorageEncoding,
  scale : @numeric_contract.ScaleGranularity,
  zero : @numeric_contract.ZeroPointMode,
  codebook : @numeric_contract.CodebookLayout,
  conversion : @numeric_contract.NumericConversion,
) -> @numeric_contract.TensorStorageContract raise {
  @numeric_contract.TensorStorageContract::new(
    dtype~,
    layout=@numeric_contract.StorageLayout::dense_row_major_v1(),
    encoding~,
    scale_granularity=scale,
    zero_point=zero,
    codebook_layout=codebook,
    conversion~,
  )
}

///|
test "storage contracts are tensor-local and metadata-reference-free" {
  let plain = test_storage(
    @numeric_contract.StorageDType::bfloat16(),
    @numeric_contract.StorageEncoding::plain_v1(),
    @numeric_contract.ScaleGranularity::absent(),
    @numeric_contract.ZeroPointMode::absent(),
    @numeric_contract.CodebookLayout::absent(),
    @numeric_contract.NumericConversion::exact_v1(),
  )
  let fp8 = test_storage(
    @numeric_contract.StorageDType::f8_e4m3_finite(),
    @numeric_contract.StorageEncoding::scaled_v1(),
    @numeric_contract.ScaleGranularity::per_tensor(),
    @numeric_contract.ZeroPointMode::absent(),
    @numeric_contract.CodebookLayout::absent(),
    @numeric_contract.NumericConversion::quantized_v1(),
  )
  let i8 = test_storage(
    @numeric_contract.StorageDType::i8(),
    @numeric_contract.StorageEncoding::symmetric_v1(),
    @numeric_contract.ScaleGranularity::per_output_channel(),
    @numeric_contract.ZeroPointMode::implicit_symmetric_zero_v1(),
    @numeric_contract.CodebookLayout::absent(),
    @numeric_contract.NumericConversion::quantized_v1(),
  )
  assert_eq(plain.digest().as_hex().length(), 64)
  assert_false(plain.digest() == fp8.digest())
  assert_false(fp8.digest() == i8.digest())
  assert_false(plain.requires_scale())
  assert_true(fp8.requires_scale())
  assert_false(i8.requires_zero_point_tensor())
}

///|
test "hostile storage-field substitutions fail closed" {
  let invalid = [
    (
      @numeric_contract.StorageDType::f8_e4m3_finite(),
      @numeric_contract.StorageEncoding::scaled_v1(),
      @numeric_contract.ScaleGranularity::absent(),
      @numeric_contract.ZeroPointMode::absent(),
      @numeric_contract.CodebookLayout::absent(),
      @numeric_contract.NumericConversion::quantized_v1(),
    ),
    (
      @numeric_contract.StorageDType::i8(),
      @numeric_contract.StorageEncoding::affine_v1(),
      @numeric_contract.ScaleGranularity::per_output_channel(),
      @numeric_contract.ZeroPointMode::absent(),
      @numeric_contract.CodebookLayout::absent(),
      @numeric_contract.NumericConversion::quantized_v1(),
    ),
    (
      @numeric_contract.StorageDType::i8(),
      @numeric_contract.StorageEncoding::codebook_v1(),
      @numeric_contract.ScaleGranularity::absent(),
      @numeric_contract.ZeroPointMode::absent(),
      @numeric_contract.CodebookLayout::absent(),
      @numeric_contract.NumericConversion::exact_v1(),
    ),
  ]
  for entry in invalid {
    let (dtype, encoding, scale, zero, codebook, conversion) = entry
    try test_storage(dtype, encoding, scale, zero, codebook, conversion) catch {
      @numeric_contract.InvalidNumericContract(StorageCombination) => ()
      _ => fail("unexpected hostile storage failure")
    } noraise {
      _ => fail("incomplete tensor storage was admitted")
    }
  }
}

///|
test "execution precision is bound independently per operation" {
  let bf16 = @numeric_contract.OperationExecutionContract::new(
    activation_input=@numeric_contract.ComputeDType::bfloat16(),
    activation_compute=@numeric_contract.ComputeDType::bfloat16(),
    activation_scale=@numeric_contract.ActivationScalePolicy::absent_exact_v1(),
    tensor_input=@numeric_contract.ComputeDType::bfloat16(),
    accumulator=@numeric_contract.AccumulatorDType::f32(),
    output=@numeric_contract.ComputeDType::bfloat16(),
    conversion=@numeric_contract.NumericConversion::exact_v1(),
    order=@numeric_contract.OperationOrderPolicy::strict_declared_order_v1(),
  )
  let fp8 = @numeric_contract.OperationExecutionContract::new(
    activation_input=@numeric_contract.ComputeDType::bfloat16(),
    activation_compute=@numeric_contract.ComputeDType::f8_e4m3_finite(),
    activation_scale=@numeric_contract.ActivationScalePolicy::dynamic_per_tensor_f32_v1(),
    tensor_input=@numeric_contract.ComputeDType::f8_e4m3_finite(),
    accumulator=@numeric_contract.AccumulatorDType::f32(),
    output=@numeric_contract.ComputeDType::bfloat16(),
    conversion=@numeric_contract.NumericConversion::quantized_v1(),
    order=@numeric_contract.OperationOrderPolicy::strict_declared_order_v1(),
  )
  assert_false(bf16.digest() == fp8.digest())
  assert_false(bf16.canonical_bytes() == fp8.canonical_bytes())
  let hostile = [
    (
      @numeric_contract.ComputeDType::f8_e4m3_finite(),
      @numeric_contract.ActivationScalePolicy::dynamic_per_tensor_f32_v1(),
      @numeric_contract.NumericConversion::exact_v1(),
    ),
    (
      @numeric_contract.ComputeDType::f8_e4m3_finite(),
      @numeric_contract.ActivationScalePolicy::absent_exact_v1(),
      @numeric_contract.NumericConversion::quantized_v1(),
    ),
    (
      @numeric_contract.ComputeDType::bfloat16(),
      @numeric_contract.ActivationScalePolicy::dynamic_per_tensor_f32_v1(),
      @numeric_contract.NumericConversion::quantized_v1(),
    ),
  ]
  for entry in hostile {
    let (compute, scale, conversion) = entry
    try
      @numeric_contract.OperationExecutionContract::new(
        activation_input=@numeric_contract.ComputeDType::bfloat16(),
        activation_compute=compute,
        activation_scale=scale,
        tensor_input=@numeric_contract.ComputeDType::f8_e4m3_finite(),
        accumulator=@numeric_contract.AccumulatorDType::f32(),
        output=@numeric_contract.ComputeDType::bfloat16(),
        conversion~,
        order=@numeric_contract.OperationOrderPolicy::strict_declared_order_v1(),
      )
    catch {
      InvalidNumericContract(ExecutionCombination) => ()
      _ => fail("unexpected hostile operation failure")
    } noraise {
      _ => fail("incomplete activation conversion was admitted")
    }
  }
}

///|
test "compound gated policy owns two exact ordered stages" {
  let simple = @numeric_contract.ActivationScalePolicy::dynamic_per_tensor_f32_v1()
  let gated = @numeric_contract.ActivationScalePolicy::gated_mlp_external_and_post_silu_f32_v1()
  let external = @numeric_contract.ActivationScaleStage::external_operation_input_v1()
  let post_silu = @numeric_contract.ActivationScaleStage::post_silu_gate_up_product_v1()
  assert_eq(simple.stage_count(), 1)
  assert_eq(simple.stage_at(0), Some(external))
  assert_eq(simple.stage_at(1), None)
  assert_eq(gated.stage_count(), 2)
  assert_eq(gated.stage_at(0), Some(external))
  assert_eq(gated.stage_at(1), Some(post_silu))
  assert_eq(gated.stage_at(-1), None)
  assert_eq(gated.stage_at(2), None)
  assert_true(simple != gated)
}

///|
test "compound gated policy is a distinct exact FP8 execution identity" {
  let simple = try! @numeric_contract.OperationExecutionContract::new(
    activation_input=@numeric_contract.ComputeDType::bfloat16(),
    activation_compute=@numeric_contract.ComputeDType::f8_e4m3_finite(),
    activation_scale=@numeric_contract.ActivationScalePolicy::dynamic_per_tensor_f32_v1(),
    tensor_input=@numeric_contract.ComputeDType::f8_e4m3_finite(),
    accumulator=@numeric_contract.AccumulatorDType::f32(),
    output=@numeric_contract.ComputeDType::bfloat16(),
    conversion=@numeric_contract.NumericConversion::quantized_v1(),
    order=@numeric_contract.OperationOrderPolicy::strict_declared_order_v1(),
  )
  let gated = try! @numeric_contract.OperationExecutionContract::new(
    activation_input=@numeric_contract.ComputeDType::bfloat16(),
    activation_compute=@numeric_contract.ComputeDType::f8_e4m3_finite(),
    activation_scale=@numeric_contract.ActivationScalePolicy::gated_mlp_external_and_post_silu_f32_v1(),
    tensor_input=@numeric_contract.ComputeDType::f8_e4m3_finite(),
    accumulator=@numeric_contract.AccumulatorDType::f32(),
    output=@numeric_contract.ComputeDType::bfloat16(),
    conversion=@numeric_contract.NumericConversion::quantized_v1(),
    order=@numeric_contract.OperationOrderPolicy::strict_declared_order_v1(),
  )
  assert_true(simple.digest() != gated.digest())
  assert_true(simple.canonical_bytes() != gated.canonical_bytes())
  assert_eq(
    try! @numeric_contract.load_operation_execution_contract(
      gated.canonical_bytes(),
      gated.digest(),
    ),
    gated,
  )
}
