///|
const MODEL_NUMERIC_SCHEMA_CANONICAL_BASE : Int = 49

///|
const MODEL_NUMERIC_SCHEMA_TENSOR_BASE : Int = 101

///|
const MODEL_NUMERIC_SCHEMA_OPERATION_SIZE : Int = 67

///|
fn optional_metadata_count(tensor : NumericTensorSchema) -> Int {
  let mut count = 0
  if tensor.scale is Some(_) {
    count += 1
  }
  if tensor.zero_point is Some(_) {
    count += 1
  }
  if tensor.codebook is Some(_) {
    count += 1
  }
  count
}

///|
fn checked_schema_size_add(
  current : Int,
  amount : Int,
  maximum : Int,
) -> Int raise NumericContractError {
  if amount < 0 || current > maximum - amount {
    raise NumericSchemaLimitExceeded(SchemaCanonicalBytes)
  }
  current + amount
}

///|
fn preflight_model_numeric_schema_size(
  limits : ModelNumericSchemaLimits,
  tensors : ArrayView[NumericTensorSchema],
  operations : ArrayView[NumericOperationSchema],
  counters : NumericSchemaConstructionCounters?,
) -> Int raise NumericContractError {
  if tensors.length() > limits.max_tensors {
    raise NumericSchemaLimitExceeded(SchemaTensorCount)
  }
  if operations.length() > limits.max_operations {
    raise NumericSchemaLimitExceeded(SchemaOperationCount)
  }
  let mut size = MODEL_NUMERIC_SCHEMA_CANONICAL_BASE
  for tensor in tensors {
    count_preflight_tensor(counters)
    let rank = tensor.shape.rank()
    if rank > limits.max_rank {
      raise NumericSchemaLimitExceeded(SchemaRank)
    }
    let entry_size = MODEL_NUMERIC_SCHEMA_TENSOR_BASE +
      rank * 8 +
      optional_metadata_count(tensor) * 8
    size = checked_schema_size_add(size, entry_size, limits.max_canonical_bytes)
  }
  let operation_bytes = operations.length() *
    MODEL_NUMERIC_SCHEMA_OPERATION_SIZE
  checked_schema_size_add(size, operation_bytes, limits.max_canonical_bytes)
}

///|
fn write_schema_optional(
  writer : NumericCanonicalWriter,
  value : NumericTensorOrdinal?,
) -> Unit raise NumericContractError {
  match value {
    Some(ordinal) => {
      writer.write_int(1)
      writer.write_int(ordinal.value)
    }
    None => writer.write_int(0)
  }
}

///|
fn write_model_numeric_schema(
  writer : NumericCanonicalWriter,
  tensors : ReadOnlyArray[NumericTensorSchema],
  operations : ReadOnlyArray[NumericOperationSchema],
) -> Unit raise NumericContractError {
  writer.write_bytes(b"lunaflux.model-numeric-schema.v2\x00")
  writer.write_int(tensors.length())
  for tensor in tensors {
    writer.write_int(tensor.ordinal.value)
    writer.write_int(tensor.role.value)
    writer.write_int(tensor.shape.rank())
    for dimension in tensor.shape.dimensions() {
      writer.write_int64(dimension)
    }
    writer.write_int(tensor.storage.canonical_length)
    write_tensor_storage_contract(writer, tensor.storage)
    write_schema_optional(
      writer,
      match tensor.scale {
        Some(value) => Some({ value: value.value })
        None => None
      },
    )
    write_schema_optional(
      writer,
      match tensor.zero_point {
        Some(value) => Some({ value: value.value })
        None => None
      },
    )
    write_schema_optional(
      writer,
      match tensor.codebook {
        Some(value) => Some({ value: value.value })
        None => None
      },
    )
  }
  writer.write_int(operations.length())
  for operation in operations {
    writer.write_int(operation.ordinal.value)
    writer.write_int(operation.execution.canonical_length)
    write_operation_execution_contract(writer, operation.execution)
  }
}

///|
fn schema_optional_byte(present : Bool, value : Int, index : Int) -> Byte {
  let length = if present { 16 } else { 8 }
  if index < 0 || index >= length {
    abort("numeric schema optional byte index out of bounds")
  }
  if index < 8 {
    numeric_canonical_int_byte(if present { 1 } else { 0 }, index)
  } else {
    numeric_canonical_int_byte(value, index - 8)
  }
}

///|
fn schema_tensor_byte_at(tensor : NumericTensorSchema, index : Int) -> Byte {
  let rank = tensor.shape.rank()
  let dimension_start = 24
  let dimension_end = dimension_start + rank * 8
  let storage_length_start = dimension_end
  let storage_start = storage_length_start + 8
  let storage_end = storage_start + tensor.storage.canonical_length
  if index < 0 {
    abort("negative numeric schema tensor byte index")
  } else if index < 8 {
    return numeric_canonical_int_byte(tensor.ordinal.value, index)
  } else if index < 16 {
    return numeric_canonical_int_byte(tensor.role.value, index - 8)
  } else if index < dimension_start {
    return numeric_canonical_int_byte(rank, index - 16)
  } else if index < dimension_end {
    let relative = index - dimension_start
    guard tensor.shape.dimension(relative / 8) is Some(dimension) else {
      abort("numeric schema dimension cursor escaped validated shape")
    }
    return numeric_canonical_int64_byte(dimension, relative % 8)
  } else if index < storage_start {
    return numeric_canonical_int_byte(
      tensor.storage.canonical_length,
      index - storage_length_start,
    )
  } else if index < storage_end {
    return tensor_storage_contract_byte_at(
      tensor.storage,
      index - storage_start,
    )
  }
  let mut optional_start = storage_end
  let scale_present = tensor.scale is Some(_)
  let scale_length = if scale_present { 16 } else { 8 }
  if index < optional_start + scale_length {
    let scale_value = match tensor.scale {
      Some(value) => value.value
      None => 0
    }
    return schema_optional_byte(
      scale_present,
      scale_value,
      index - optional_start,
    )
  }
  optional_start += scale_length
  let zero_present = tensor.zero_point is Some(_)
  let zero_length = if zero_present { 16 } else { 8 }
  if index < optional_start + zero_length {
    let zero_value = match tensor.zero_point {
      Some(value) => value.value
      None => 0
    }
    return schema_optional_byte(
      zero_present,
      zero_value,
      index - optional_start,
    )
  }
  optional_start += zero_length
  let codebook_present = tensor.codebook is Some(_)
  let codebook_length = if codebook_present { 16 } else { 8 }
  if index < optional_start + codebook_length {
    let codebook_value = match tensor.codebook {
      Some(value) => value.value
      None => 0
    }
    return schema_optional_byte(
      codebook_present,
      codebook_value,
      index - optional_start,
    )
  }
  abort("numeric schema tensor byte index out of bounds")
}

///|
fn schema_tensor_canonical_length(tensor : NumericTensorSchema) -> Int {
  MODEL_NUMERIC_SCHEMA_TENSOR_BASE +
  tensor.shape.rank() * 8 +
  optional_metadata_count(tensor) * 8
}

///|
fn schema_operation_byte_at(
  operation : NumericOperationSchema,
  index : Int,
) -> Byte {
  if index < 0 {
    abort("negative numeric schema operation byte index")
  } else if index < 8 {
    numeric_canonical_int_byte(operation.ordinal.value, index)
  } else if index < 16 {
    numeric_canonical_int_byte(operation.execution.canonical_length, index - 8)
  } else if index < MODEL_NUMERIC_SCHEMA_OPERATION_SIZE {
    operation_execution_contract_byte_at(operation.execution, index - 16)
  } else {
    abort("numeric schema operation byte index out of bounds")
  }
}

///|
priv struct ModelNumericSchemaCanonicalCursor {
  tensors : ReadOnlyArray[NumericTensorSchema]
  operations : ReadOnlyArray[NumericOperationSchema]
  mut tensor_index : Int
  mut tensor_start : Int
}

///|
fn ModelNumericSchemaCanonicalCursor::new(
  tensors : ReadOnlyArray[NumericTensorSchema],
  operations : ReadOnlyArray[NumericOperationSchema],
) -> ModelNumericSchemaCanonicalCursor {
  { tensors, operations, tensor_index: 0, tensor_start: 41 }
}

///|
fn ModelNumericSchemaCanonicalCursor::byte_at(
  self : ModelNumericSchemaCanonicalCursor,
  index : Int,
) -> Byte {
  let domain = b"lunaflux.model-numeric-schema.v2\x00"
  if index < domain.length() {
    return domain[index]
  } else if index < 41 {
    return numeric_canonical_int_byte(self.tensors.length(), index - 33)
  }
  while self.tensor_index < self.tensors.length() {
    let tensor = self.tensors[self.tensor_index]
    let tensor_end = self.tensor_start + schema_tensor_canonical_length(tensor)
    if index < tensor_end {
      return schema_tensor_byte_at(tensor, index - self.tensor_start)
    }
    self.tensor_start = tensor_end
    self.tensor_index += 1
  }
  if index < self.tensor_start + 8 {
    return numeric_canonical_int_byte(
      self.operations.length(),
      index - self.tensor_start,
    )
  }
  let operation_offset = index - self.tensor_start - 8
  let operation_index = operation_offset / MODEL_NUMERIC_SCHEMA_OPERATION_SIZE
  if operation_index < 0 || operation_index >= self.operations.length() {
    abort("numeric schema operation cursor escaped validated operations")
  }
  schema_operation_byte_at(
    self.operations[operation_index],
    operation_offset % MODEL_NUMERIC_SCHEMA_OPERATION_SIZE,
  )
}

///|
fn digest_model_numeric_schema(
  canonical_length : Int,
  tensors : ReadOnlyArray[NumericTensorSchema],
  operations : ReadOnlyArray[NumericOperationSchema],
) -> ModelNumericSchemaDigest {
  let writer = NumericCanonicalWriter::new_hashing(canonical_length)
  try! write_model_numeric_schema(writer, tensors, operations)
  { value: try! writer.finish_hex() }
}

///|
fn ModelNumericSchema::encode_canonical_bytes(
  self : ModelNumericSchema,
) -> Bytes {
  let cursor = ModelNumericSchemaCanonicalCursor::new(
    self.tensors,
    self.operations,
  )
  let bytes = materialize_numeric_canonical(self.canonical_length, index => {
    cursor.byte_at(index)
  })
  if digest_bytes(bytes) != self.digest.value {
    abort("numeric schema canonical materialization diverged from its digest")
  }
  bytes
}
