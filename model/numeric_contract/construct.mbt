///|
pub fn NumericConversion::new(
  rounding~ : RoundingPolicy,
  saturation~ : SaturationPolicy,
  finite_values~ : FiniteValuePolicy,
) -> NumericConversion raise NumericContractError {
  let exact = rounding == RoundingPolicy::exact_no_rounding_v1() &&
    saturation == SaturationPolicy::none_v1()
  let quantized = rounding == RoundingPolicy::nearest_ties_to_even_v1() &&
    saturation == SaturationPolicy::finite_storage_range_v1()
  if finite_values != FiniteValuePolicy::reject_non_finite_v1() ||
    !(exact || quantized) {
    raise InvalidNumericContract(UnsupportedValue)
  }
  { rounding, saturation, finite_values }
}

///|
pub fn NumericConversion::exact_v1() -> NumericConversion {
  try! NumericConversion::new(
    rounding=RoundingPolicy::exact_no_rounding_v1(),
    saturation=SaturationPolicy::none_v1(),
    finite_values=FiniteValuePolicy::reject_non_finite_v1(),
  )
}

///|
pub fn NumericConversion::quantized_v1() -> NumericConversion {
  try! NumericConversion::new(
    rounding=RoundingPolicy::nearest_ties_to_even_v1(),
    saturation=SaturationPolicy::finite_storage_range_v1(),
    finite_values=FiniteValuePolicy::reject_non_finite_v1(),
  )
}

///|
fn known_storage_dtype(dtype : StorageDType) -> Bool {
  dtype == StorageDType::bfloat16() ||
  dtype == StorageDType::f8_e4m3_finite() ||
  dtype == StorageDType::i8() ||
  dtype == StorageDType::f32()
}

///|
fn valid_storage_combination(
  dtype : StorageDType,
  encoding : StorageEncoding,
  scale : ScaleGranularity,
  zero_point : ZeroPointMode,
  codebook : CodebookLayout,
  conversion : NumericConversion,
) -> Bool {
  let plain = encoding == StorageEncoding::plain_v1() &&
    scale == ScaleGranularity::absent() &&
    zero_point == ZeroPointMode::absent() &&
    codebook == CodebookLayout::absent() &&
    conversion == NumericConversion::exact_v1()
  let scaled = encoding == StorageEncoding::scaled_v1() &&
    dtype == StorageDType::f8_e4m3_finite() &&
    scale == ScaleGranularity::per_tensor() &&
    zero_point == ZeroPointMode::absent() &&
    codebook == CodebookLayout::absent() &&
    conversion == NumericConversion::quantized_v1()
  let symmetric = encoding == StorageEncoding::symmetric_v1() &&
    dtype == StorageDType::i8() &&
    scale == ScaleGranularity::per_output_channel() &&
    zero_point == ZeroPointMode::implicit_symmetric_zero_v1() &&
    codebook == CodebookLayout::absent() &&
    conversion == NumericConversion::quantized_v1()
  let affine = encoding == StorageEncoding::affine_v1() &&
    dtype == StorageDType::i8() &&
    scale == ScaleGranularity::per_output_channel() &&
    zero_point == ZeroPointMode::per_output_channel_tensor_v1() &&
    codebook == CodebookLayout::absent() &&
    conversion == NumericConversion::quantized_v1()
  let codebook_encoded = encoding == StorageEncoding::codebook_v1() &&
    dtype == StorageDType::i8() &&
    scale == ScaleGranularity::absent() &&
    zero_point == ZeroPointMode::absent() &&
    codebook == CodebookLayout::contiguous_f32_values_v1() &&
    conversion == NumericConversion::exact_v1()
  plain || scaled || symmetric || affine || codebook_encoded
}

///|
/// Constructs the reference-free representation contract for one tensor.
pub fn TensorStorageContract::new(
  dtype~ : StorageDType,
  layout~ : StorageLayout,
  encoding~ : StorageEncoding,
  scale_granularity~ : ScaleGranularity,
  zero_point~ : ZeroPointMode,
  codebook_layout~ : CodebookLayout,
  conversion~ : NumericConversion,
) -> TensorStorageContract raise NumericContractError {
  if !known_storage_dtype(dtype) ||
    layout != StorageLayout::dense_row_major_v1() ||
    !valid_storage_combination(
      dtype, encoding, scale_granularity, zero_point, codebook_layout, conversion,
    ) {
    raise InvalidNumericContract(StorageCombination)
  }
  build_tensor_storage_contract(
    dtype, layout, encoding, scale_granularity, zero_point, codebook_layout, conversion,
  )
}

///|
fn known_compute_dtype(dtype : ComputeDType) -> Bool {
  dtype == ComputeDType::absent() ||
  dtype == ComputeDType::bfloat16() ||
  dtype == ComputeDType::f8_e4m3_finite() ||
  dtype == ComputeDType::i8() ||
  dtype == ComputeDType::f32()
}

///|
/// Constructs the execution contract for exactly one operation.
pub fn OperationExecutionContract::new(
  activation_input~ : ComputeDType,
  activation_compute~ : ComputeDType,
  activation_scale~ : ActivationScalePolicy,
  tensor_input~ : ComputeDType,
  accumulator~ : AccumulatorDType,
  output~ : ComputeDType,
  conversion~ : NumericConversion,
  order~ : OperationOrderPolicy,
) -> OperationExecutionContract raise NumericContractError {
  let inputs_present = activation_input != ComputeDType::absent() ||
    tensor_input != ComputeDType::absent()
  let valid_accumulator = accumulator == AccumulatorDType::f32() ||
    accumulator == AccumulatorDType::i32()
  let exact_activation = activation_compute == activation_input &&
    activation_scale == ActivationScalePolicy::absent_exact_v1() &&
    conversion == NumericConversion::exact_v1()
  let finite_fp8_activation = activation_input == ComputeDType::bfloat16() &&
    activation_compute == ComputeDType::f8_e4m3_finite() &&
    (
      activation_scale == ActivationScalePolicy::dynamic_per_tensor_f32_v1() ||
      activation_scale ==
      ActivationScalePolicy::gated_mlp_external_and_post_silu_f32_v1() ||
      activation_scale ==
      ActivationScalePolicy::gated_mlp_external_and_post_silu_bound_expf_f32_v2()
    ) &&
    conversion == NumericConversion::quantized_v1() &&
    tensor_input == ComputeDType::f8_e4m3_finite() &&
    accumulator == AccumulatorDType::f32() &&
    output == ComputeDType::bfloat16()
  let activation_presence_matches = (activation_input == ComputeDType::absent()) ==
    (activation_compute == ComputeDType::absent())
  if !known_compute_dtype(activation_input) ||
    !known_compute_dtype(activation_compute) ||
    !known_compute_dtype(tensor_input) ||
    !known_compute_dtype(output) ||
    output == ComputeDType::absent() ||
    !inputs_present ||
    !valid_accumulator ||
    !activation_presence_matches ||
    !(exact_activation || finite_fp8_activation) ||
    order != OperationOrderPolicy::strict_declared_order_v1() {
    raise InvalidNumericContract(ExecutionCombination)
  }
  build_operation_execution_contract(
    activation_input, activation_compute, activation_scale, tensor_input, accumulator,
    output, conversion, order,
  )
}
