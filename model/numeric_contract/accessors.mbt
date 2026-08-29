///|
pub fn NumericConversion::rounding(self : NumericConversion) -> RoundingPolicy {
  self.rounding
}

///|
pub fn NumericConversion::saturation(
  self : NumericConversion,
) -> SaturationPolicy {
  self.saturation
}

///|
pub fn NumericConversion::finite_values(
  self : NumericConversion,
) -> FiniteValuePolicy {
  self.finite_values
}

///|
pub fn TensorStorageContract::dtype(
  self : TensorStorageContract,
) -> StorageDType {
  self.dtype
}

///|
pub fn TensorStorageContract::layout(
  self : TensorStorageContract,
) -> StorageLayout {
  self.layout
}

///|
pub fn TensorStorageContract::encoding(
  self : TensorStorageContract,
) -> StorageEncoding {
  self.encoding
}

///|
pub fn TensorStorageContract::scale_granularity(
  self : TensorStorageContract,
) -> ScaleGranularity {
  self.scale_granularity
}

///|
pub fn TensorStorageContract::zero_point(
  self : TensorStorageContract,
) -> ZeroPointMode {
  self.zero_point
}

///|
pub fn TensorStorageContract::codebook_layout(
  self : TensorStorageContract,
) -> CodebookLayout {
  self.codebook_layout
}

///|
pub fn TensorStorageContract::conversion(
  self : TensorStorageContract,
) -> NumericConversion {
  self.conversion
}

///|
pub fn TensorStorageContract::canonical_bytes(
  self : TensorStorageContract,
) -> Bytes {
  encode_tensor_storage_contract(self)
}

///|
pub fn TensorStorageContract::digest(
  self : TensorStorageContract,
) -> TensorStorageDigest {
  self.digest
}

///|
pub fn TensorStorageContract::requires_scale(
  self : TensorStorageContract,
) -> Bool {
  self.scale_granularity != ScaleGranularity::absent()
}

///|
pub fn TensorStorageContract::requires_zero_point_tensor(
  self : TensorStorageContract,
) -> Bool {
  self.zero_point == ZeroPointMode::per_output_channel_tensor_v1()
}

///|
pub fn TensorStorageContract::requires_codebook(
  self : TensorStorageContract,
) -> Bool {
  self.codebook_layout != CodebookLayout::absent()
}

///|
pub fn OperationExecutionContract::activation_input(
  self : OperationExecutionContract,
) -> ComputeDType {
  self.activation_input
}

///|
pub fn OperationExecutionContract::activation_compute(
  self : OperationExecutionContract,
) -> ComputeDType {
  self.activation_compute
}

///|
pub fn OperationExecutionContract::activation_scale(
  self : OperationExecutionContract,
) -> ActivationScalePolicy {
  self.activation_scale
}

///|
pub fn OperationExecutionContract::tensor_input(
  self : OperationExecutionContract,
) -> ComputeDType {
  self.tensor_input
}

///|
pub fn OperationExecutionContract::accumulator(
  self : OperationExecutionContract,
) -> AccumulatorDType {
  self.accumulator
}

///|
pub fn OperationExecutionContract::output(
  self : OperationExecutionContract,
) -> ComputeDType {
  self.output
}

///|
pub fn OperationExecutionContract::conversion(
  self : OperationExecutionContract,
) -> NumericConversion {
  self.conversion
}

///|
pub fn OperationExecutionContract::order(
  self : OperationExecutionContract,
) -> OperationOrderPolicy {
  self.order
}

///|
pub fn OperationExecutionContract::canonical_bytes(
  self : OperationExecutionContract,
) -> Bytes {
  encode_operation_execution_contract(self)
}

///|
pub fn OperationExecutionContract::digest(
  self : OperationExecutionContract,
) -> OperationExecutionDigest {
  self.digest
}

///|
pub fn OperationExecutionContract::accepts_storage(
  self : OperationExecutionContract,
  storage : TensorStorageContract,
) -> Bool {
  match self.tensor_input.value {
    1 =>
      storage.dtype == StorageDType::bfloat16() &&
      storage.encoding == StorageEncoding::plain_v1()
    2 =>
      storage.dtype == StorageDType::f8_e4m3_finite() &&
      storage.encoding == StorageEncoding::scaled_v1()
    3 =>
      storage.dtype == StorageDType::i8() &&
      storage.encoding == StorageEncoding::symmetric_v1()
    4 =>
      storage.dtype == StorageDType::f32() &&
      storage.encoding == StorageEncoding::plain_v1()
    _ => false
  }
}
