///|
pub fn NumericTensorOrdinal::new(
  value : Int,
) -> NumericTensorOrdinal raise NumericContractError {
  if value < 0 {
    raise InvalidNumericContract(Identifier)
  }
  { value, }
}

///|
pub fn NumericTensorOrdinal::as_int(self : NumericTensorOrdinal) -> Int {
  self.value
}

///|
pub fn NumericOperationOrdinal::new(
  value : Int,
) -> NumericOperationOrdinal raise NumericContractError {
  if value < 0 {
    raise InvalidNumericContract(Identifier)
  }
  { value, }
}

///|
pub fn NumericOperationOrdinal::as_int(self : NumericOperationOrdinal) -> Int {
  self.value
}

///|
pub fn ScaleTensorOrdinal::new(
  tensor : NumericTensorOrdinal,
) -> ScaleTensorOrdinal {
  { value: tensor.value }
}

///|
pub fn ScaleTensorOrdinal::tensor(
  self : ScaleTensorOrdinal,
) -> NumericTensorOrdinal {
  { value: self.value }
}

///|
pub fn ZeroPointTensorOrdinal::new(
  tensor : NumericTensorOrdinal,
) -> ZeroPointTensorOrdinal {
  { value: tensor.value }
}

///|
pub fn ZeroPointTensorOrdinal::tensor(
  self : ZeroPointTensorOrdinal,
) -> NumericTensorOrdinal {
  { value: self.value }
}

///|
pub fn CodebookTensorOrdinal::new(
  tensor : NumericTensorOrdinal,
) -> CodebookTensorOrdinal {
  { value: tensor.value }
}

///|
pub fn CodebookTensorOrdinal::tensor(
  self : CodebookTensorOrdinal,
) -> NumericTensorOrdinal {
  { value: self.value }
}

///|
pub fn NumericTensorRole::parameter() -> NumericTensorRole {
  { value: 1 }
}

///|
pub fn NumericTensorRole::scale_metadata() -> NumericTensorRole {
  { value: 2 }
}

///|
pub fn NumericTensorRole::zero_point_metadata() -> NumericTensorRole {
  { value: 3 }
}

///|
pub fn NumericTensorRole::codebook_metadata() -> NumericTensorRole {
  { value: 4 }
}

///|
pub fn NumericTensorSchema::new(
  ordinal~ : NumericTensorOrdinal,
  role~ : NumericTensorRole,
  shape~ : NumericShape,
  storage~ : TensorStorageContract,
  scale? : ScaleTensorOrdinal,
  zero_point? : ZeroPointTensorOrdinal,
  codebook? : CodebookTensorOrdinal,
) -> NumericTensorSchema raise NumericContractError {
  let known_role = role == NumericTensorRole::parameter() ||
    role == NumericTensorRole::scale_metadata() ||
    role == NumericTensorRole::zero_point_metadata() ||
    role == NumericTensorRole::codebook_metadata()
  if !known_role ||
    storage.requires_scale() != (scale is Some(_)) ||
    storage.requires_zero_point_tensor() != (zero_point is Some(_)) ||
    storage.requires_codebook() != (codebook is Some(_)) {
    raise InvalidNumericContract(StorageCombination)
  }
  { ordinal, role, shape, storage, scale, zero_point, codebook }
}

///|
pub fn NumericOperationSchema::new(
  ordinal~ : NumericOperationOrdinal,
  execution~ : OperationExecutionContract,
) -> NumericOperationSchema {
  { ordinal, execution }
}

///|
pub fn ModelNumericSchema::new(
  limits~ : ModelNumericSchemaLimits,
  tensors~ : ArrayView[NumericTensorSchema],
  operations~ : ArrayView[NumericOperationSchema],
) -> ModelNumericSchema raise NumericContractError {
  let counters = new_numeric_schema_construction_counters()
  new_model_numeric_schema_with_counters(
    limits,
    tensors,
    operations,
    Some(counters),
  )
}

///|
pub fn NumericTensorSchema::ordinal(
  self : NumericTensorSchema,
) -> NumericTensorOrdinal {
  self.ordinal
}

///|
pub fn NumericTensorSchema::role(
  self : NumericTensorSchema,
) -> NumericTensorRole {
  self.role
}

///|
pub fn NumericTensorSchema::shape(self : NumericTensorSchema) -> NumericShape {
  self.shape
}

///|
pub fn NumericTensorSchema::storage(
  self : NumericTensorSchema,
) -> TensorStorageContract {
  self.storage
}

///|
pub fn NumericTensorSchema::scale(
  self : NumericTensorSchema,
) -> ScaleTensorOrdinal? {
  self.scale
}

///|
pub fn NumericTensorSchema::zero_point(
  self : NumericTensorSchema,
) -> ZeroPointTensorOrdinal? {
  self.zero_point
}

///|
pub fn NumericTensorSchema::codebook(
  self : NumericTensorSchema,
) -> CodebookTensorOrdinal? {
  self.codebook
}

///|
pub fn NumericOperationSchema::ordinal(
  self : NumericOperationSchema,
) -> NumericOperationOrdinal {
  self.ordinal
}

///|
pub fn NumericOperationSchema::execution(
  self : NumericOperationSchema,
) -> OperationExecutionContract {
  self.execution
}

///|
pub fn ModelNumericSchema::tensors(
  self : ModelNumericSchema,
) -> ReadOnlyArray[NumericTensorSchema] {
  self.tensors
}

///|
pub fn ModelNumericSchema::operations(
  self : ModelNumericSchema,
) -> ReadOnlyArray[NumericOperationSchema] {
  self.operations
}

///|
pub fn ModelNumericSchema::canonical_bytes(self : ModelNumericSchema) -> Bytes {
  self.encode_canonical_bytes()
}

///|
pub fn ModelNumericSchema::digest(
  self : ModelNumericSchema,
) -> ModelNumericSchemaDigest {
  self.digest
}

///|
pub fn ModelNumericSchemaDigest::from_sha256(
  value : String,
) -> ModelNumericSchemaDigest raise NumericContractError {
  if !is_lower_sha256(value) {
    raise InvalidNumericContract(Digest)
  }
  { value, }
}

///|
pub fn ModelNumericSchemaDigest::as_hex(
  self : ModelNumericSchemaDigest,
) -> String {
  self.value
}
