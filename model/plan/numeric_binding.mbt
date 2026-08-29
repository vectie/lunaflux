///|
/// Generic role of one entry in the immutable model tensor table.
pub(all) enum PlanTensorRole {
  ParameterTensor
  ScaleMetadataTensor
  ZeroPointMetadataTensor
  CodebookMetadataTensor
} derive(Debug, Eq)

///|
/// One typed tensor-table entry mapped from the canonical numeric schema.
pub struct PlanTensor {
  priv reference : TensorRef
  priv role : PlanTensorRole
  priv shape : @numeric_contract.NumericShape
  priv storage : @numeric_contract.TensorStorageContract
  priv scale : ScaleTensorRef?
  priv zero_point : ZeroPointTensorRef?
  priv codebook : CodebookTensorRef?
} derive(Debug, Eq)

///|
/// Typed operation identity mapped from one canonical numeric schema entry.
pub struct OperationNumericBinding {
  priv operation : OperationId
  priv execution : @numeric_contract.OperationExecutionContract
} derive(Debug, Eq)

///|
/// Typed view of the one canonical numeric schema embedded in `ModelPlan`.
///
/// This value owns no independent encoder or digest.
pub struct ModelNumericBinding {
  priv schema : @numeric_contract.ModelNumericSchema
  priv tensors : ReadOnlyArray[PlanTensor]
  priv operations : ReadOnlyArray[OperationNumericBinding]
} derive(Debug, Eq)

///|
fn mapped_role(
  role : @numeric_contract.NumericTensorRole,
) -> PlanTensorRole raise PlanValidationError {
  if role == @numeric_contract.NumericTensorRole::parameter() {
    ParameterTensor
  } else if role == @numeric_contract.NumericTensorRole::scale_metadata() {
    ScaleMetadataTensor
  } else if role == @numeric_contract.NumericTensorRole::zero_point_metadata() {
    ZeroPointMetadataTensor
  } else if role == @numeric_contract.NumericTensorRole::codebook_metadata() {
    CodebookMetadataTensor
  } else {
    raise InvalidGraph(TensorRoleMismatch)
  }
}

///|
fn mapped_tensor_ref(
  ordinal : @numeric_contract.NumericTensorOrdinal,
) -> TensorRef raise PlanValidationError {
  TensorRef::new(ordinal.as_int())
}

///|
fn mapped_scale_ref(
  reference : @numeric_contract.ScaleTensorOrdinal?,
) -> ScaleTensorRef? raise PlanValidationError {
  match reference {
    Some(value) => Some(ScaleTensorRef::new(mapped_tensor_ref(value.tensor())))
    None => None
  }
}

///|
fn mapped_zero_point_ref(
  reference : @numeric_contract.ZeroPointTensorOrdinal?,
) -> ZeroPointTensorRef? raise PlanValidationError {
  match reference {
    Some(value) =>
      Some(ZeroPointTensorRef::new(mapped_tensor_ref(value.tensor())))
    None => None
  }
}

///|
fn mapped_codebook_ref(
  reference : @numeric_contract.CodebookTensorOrdinal?,
) -> CodebookTensorRef? raise PlanValidationError {
  match reference {
    Some(value) =>
      Some(CodebookTensorRef::new(mapped_tensor_ref(value.tensor())))
    None => None
  }
}

///|
fn map_numeric_schema(
  schema : @numeric_contract.ModelNumericSchema,
) -> ModelNumericBinding raise PlanValidationError {
  let tensors : Array[PlanTensor] = []
  for tensor in schema.tensors() {
    tensors.push({
      reference: mapped_tensor_ref(tensor.ordinal()),
      role: mapped_role(tensor.role()),
      shape: tensor.shape(),
      storage: tensor.storage(),
      scale: mapped_scale_ref(tensor.scale()),
      zero_point: mapped_zero_point_ref(tensor.zero_point()),
      codebook: mapped_codebook_ref(tensor.codebook()),
    })
  }
  let operations : Array[OperationNumericBinding] = []
  for operation in schema.operations() {
    operations.push({
      operation: OperationId::new(operation.ordinal().as_int()),
      execution: operation.execution(),
    })
  }
  {
    schema,
    tensors: ReadOnlyArray::from_array(tensors),
    operations: ReadOnlyArray::from_array(operations),
  }
}

///|
pub fn PlanTensor::reference(self : PlanTensor) -> TensorRef {
  self.reference
}

///|
pub fn PlanTensor::role(self : PlanTensor) -> PlanTensorRole {
  self.role
}

///|
pub fn PlanTensor::shape(self : PlanTensor) -> @numeric_contract.NumericShape {
  self.shape
}

///|
pub fn PlanTensor::storage(
  self : PlanTensor,
) -> @numeric_contract.TensorStorageContract {
  self.storage
}

///|
pub fn PlanTensor::scale(self : PlanTensor) -> ScaleTensorRef? {
  self.scale
}

///|
pub fn PlanTensor::zero_point(self : PlanTensor) -> ZeroPointTensorRef? {
  self.zero_point
}

///|
pub fn PlanTensor::codebook(self : PlanTensor) -> CodebookTensorRef? {
  self.codebook
}

///|
pub fn OperationNumericBinding::operation(
  self : OperationNumericBinding,
) -> OperationId {
  self.operation
}

///|
pub fn OperationNumericBinding::execution(
  self : OperationNumericBinding,
) -> @numeric_contract.OperationExecutionContract {
  self.execution
}

///|
pub fn ModelNumericBinding::schema(
  self : ModelNumericBinding,
) -> @numeric_contract.ModelNumericSchema {
  self.schema
}

///|
pub fn ModelNumericBinding::tensors(
  self : ModelNumericBinding,
) -> ReadOnlyArray[PlanTensor] {
  self.tensors
}

///|
pub fn ModelNumericBinding::operations(
  self : ModelNumericBinding,
) -> ReadOnlyArray[OperationNumericBinding] {
  self.operations
}

///|
pub fn ModelNumericBinding::canonical_bytes(
  self : ModelNumericBinding,
) -> Bytes {
  self.schema.canonical_bytes()
}

///|
pub fn ModelNumericBinding::digest(
  self : ModelNumericBinding,
) -> @numeric_contract.ModelNumericSchemaDigest {
  self.schema.digest()
}
