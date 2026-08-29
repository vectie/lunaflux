///|
/// Constructs the exact symmetric-I8 weight representation admitted by
/// `symmetric_i8_weight_only_v1`.
///
/// The stored matrix is dense row-major `[out_channels, in_channels]`. It owns
/// one rank-one plain-F32 scale tensor indexed by `out_channels`; no explicit
/// zero-point or codebook is legal.
pub fn TensorStorageContract::symmetric_i8_weight_only_v1() -> TensorStorageContract {
  try! TensorStorageContract::new(
    dtype=StorageDType::i8(),
    layout=StorageLayout::dense_row_major_v1(),
    encoding=StorageEncoding::symmetric_v1(),
    scale_granularity=ScaleGranularity::per_output_channel(),
    zero_point=ZeroPointMode::implicit_symmetric_zero_v1(),
    codebook_layout=CodebookLayout::absent(),
    conversion=NumericConversion::quantized_v1(),
  )
}

///|
/// Constructs the exact execution side of symmetric-I8 weight-only v1.
/// Activations remain BF16 at the graph and compute boundaries; only stored
/// weights are I8. Accumulation is F32 and graph output is BF16.
pub fn OperationExecutionContract::symmetric_i8_weight_only_v1() -> OperationExecutionContract {
  try! OperationExecutionContract::new(
    activation_input=ComputeDType::bfloat16(),
    activation_compute=ComputeDType::bfloat16(),
    activation_scale=ActivationScalePolicy::absent_exact_v1(),
    tensor_input=ComputeDType::i8(),
    accumulator=AccumulatorDType::f32(),
    output=ComputeDType::bfloat16(),
    conversion=NumericConversion::exact_v1(),
    order=OperationOrderPolicy::strict_declared_order_v1(),
  )
}

///|
/// Constructs the only storage contract legal for an I8 v1 scale tensor.
pub fn TensorStorageContract::symmetric_i8_weight_only_scale_v1() -> TensorStorageContract {
  try! TensorStorageContract::new(
    dtype=StorageDType::f32(),
    layout=StorageLayout::dense_row_major_v1(),
    encoding=StorageEncoding::plain_v1(),
    scale_granularity=ScaleGranularity::absent(),
    zero_point=ZeroPointMode::absent(),
    codebook_layout=CodebookLayout::absent(),
    conversion=NumericConversion::exact_v1(),
  )
}

///|
/// Returns whether one signed integer is a legal persisted v1 weight code.
/// `-128` is deliberately reserved and rejected by this version.
pub fn symmetric_i8_weight_only_v1_accepts_code(code : Int) -> Bool {
  code >= -127 && code <= 127
}

///|
fn TensorStorageContract::is_symmetric_i8_weight_only_v1(
  self : TensorStorageContract,
) -> Bool {
  self == TensorStorageContract::symmetric_i8_weight_only_v1()
}
