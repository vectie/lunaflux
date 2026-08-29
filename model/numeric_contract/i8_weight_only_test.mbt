///|
fn i8_test_storage(
  encoding : @numeric_contract.StorageEncoding,
  scale : @numeric_contract.ScaleGranularity,
  zero_point : @numeric_contract.ZeroPointMode,
  codebook : @numeric_contract.CodebookLayout,
  conversion : @numeric_contract.NumericConversion,
) -> @numeric_contract.TensorStorageContract raise {
  @numeric_contract.TensorStorageContract::new(
    dtype=@numeric_contract.StorageDType::i8(),
    layout=@numeric_contract.StorageLayout::dense_row_major_v1(),
    encoding~,
    scale_granularity=scale,
    zero_point~,
    codebook_layout=codebook,
    conversion~,
  )
}

///|
test "symmetric I8 weight-only v1 freezes exact storage and execution fields" {
  let storage = @numeric_contract.TensorStorageContract::symmetric_i8_weight_only_v1()
  assert_eq(storage.dtype(), @numeric_contract.StorageDType::i8())
  assert_eq(
    storage.encoding(),
    @numeric_contract.StorageEncoding::symmetric_v1(),
  )
  assert_eq(
    storage.scale_granularity(),
    @numeric_contract.ScaleGranularity::per_output_channel(),
  )
  assert_eq(
    storage.zero_point(),
    @numeric_contract.ZeroPointMode::implicit_symmetric_zero_v1(),
  )
  assert_false(storage.requires_zero_point_tensor())
  assert_false(storage.requires_codebook())
  let execution = @numeric_contract.OperationExecutionContract::symmetric_i8_weight_only_v1()
  assert_eq(
    execution.activation_input(),
    @numeric_contract.ComputeDType::bfloat16(),
  )
  assert_eq(
    execution.activation_compute(),
    @numeric_contract.ComputeDType::bfloat16(),
  )
  assert_eq(
    execution.activation_scale(),
    @numeric_contract.ActivationScalePolicy::absent_exact_v1(),
  )
  assert_eq(execution.tensor_input(), @numeric_contract.ComputeDType::i8())
  assert_eq(execution.accumulator(), @numeric_contract.AccumulatorDType::f32())
  assert_eq(execution.output(), @numeric_contract.ComputeDType::bfloat16())
  assert_true(execution.accepts_storage(storage))
}

///|
test "symmetric I8 v1 reserves minus 128 and admits only signed symmetric codes" {
  assert_false(@numeric_contract.symmetric_i8_weight_only_v1_accepts_code(-128))
  assert_true(@numeric_contract.symmetric_i8_weight_only_v1_accepts_code(-127))
  assert_true(@numeric_contract.symmetric_i8_weight_only_v1_accepts_code(0))
  assert_true(@numeric_contract.symmetric_i8_weight_only_v1_accepts_code(127))
  assert_false(@numeric_contract.symmetric_i8_weight_only_v1_accepts_code(128))
}

///|
test "symmetric I8 execution rejects affine zero points and codebooks" {
  let affine = i8_test_storage(
    @numeric_contract.StorageEncoding::affine_v1(),
    @numeric_contract.ScaleGranularity::per_output_channel(),
    @numeric_contract.ZeroPointMode::per_output_channel_tensor_v1(),
    @numeric_contract.CodebookLayout::absent(),
    @numeric_contract.NumericConversion::quantized_v1(),
  )
  let codebook = i8_test_storage(
    @numeric_contract.StorageEncoding::codebook_v1(),
    @numeric_contract.ScaleGranularity::absent(),
    @numeric_contract.ZeroPointMode::absent(),
    @numeric_contract.CodebookLayout::contiguous_f32_values_v1(),
    @numeric_contract.NumericConversion::exact_v1(),
  )
  let execution = @numeric_contract.OperationExecutionContract::symmetric_i8_weight_only_v1()
  assert_false(execution.accepts_storage(affine))
  assert_false(execution.accepts_storage(codebook))
  try
    i8_test_storage(
      @numeric_contract.StorageEncoding::symmetric_v1(),
      @numeric_contract.ScaleGranularity::per_output_channel(),
      @numeric_contract.ZeroPointMode::per_output_channel_tensor_v1(),
      @numeric_contract.CodebookLayout::absent(),
      @numeric_contract.NumericConversion::quantized_v1(),
    )
  catch {
    @numeric_contract.InvalidNumericContract(StorageCombination) => ()
    _ => fail("unexpected explicit-zero-point failure")
  } noraise {
    _ => fail("explicit zero-point tensor was admitted as symmetric I8 v1")
  }
}

///|
test "FP8 activation conversion cannot be paired with I8 tensor input" {
  try
    @numeric_contract.OperationExecutionContract::new(
      activation_input=@numeric_contract.ComputeDType::bfloat16(),
      activation_compute=@numeric_contract.ComputeDType::f8_e4m3_finite(),
      activation_scale=@numeric_contract.ActivationScalePolicy::dynamic_per_tensor_f32_v1(),
      tensor_input=@numeric_contract.ComputeDType::i8(),
      accumulator=@numeric_contract.AccumulatorDType::f32(),
      output=@numeric_contract.ComputeDType::bfloat16(),
      conversion=@numeric_contract.NumericConversion::quantized_v1(),
      order=@numeric_contract.OperationOrderPolicy::strict_declared_order_v1(),
    )
  catch {
    InvalidNumericContract(ExecutionCombination) => ()
    _ => fail("unexpected mixed FP8/I8 execution failure")
  } noraise {
    _ => fail("FP8 activation conversion was admitted for I8 weight-only v1")
  }
}
