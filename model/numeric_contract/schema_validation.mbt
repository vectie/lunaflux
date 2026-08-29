///|
priv struct NumericSchemaConstructionCounters {
  mut preflight_tensors : Int
  mut owner_table_allocations : Int
  mut metadata_lookups : Int
  mut owner_claims : Int
  mut copied_entries : Int
  mut hash_walks : Int
}

///|
fn new_numeric_schema_construction_counters() -> NumericSchemaConstructionCounters {
  {
    preflight_tensors: 0,
    owner_table_allocations: 0,
    metadata_lookups: 0,
    owner_claims: 0,
    copied_entries: 0,
    hash_walks: 0,
  }
}

///|
fn count_preflight_tensor(
  counters : NumericSchemaConstructionCounters?,
) -> Unit {
  match counters {
    Some(value) => value.preflight_tensors += 1
    None => ()
  }
}

///|
fn count_owner_tables(counters : NumericSchemaConstructionCounters?) -> Unit {
  match counters {
    Some(value) => value.owner_table_allocations += 3
    None => ()
  }
}

///|
fn count_metadata_lookup(counters : NumericSchemaConstructionCounters?) -> Unit {
  match counters {
    Some(value) => value.metadata_lookups += 1
    None => ()
  }
}

///|
fn count_owner_claim(counters : NumericSchemaConstructionCounters?) -> Unit {
  match counters {
    Some(value) => value.owner_claims += 1
    None => ()
  }
}

///|
fn schema_plain_metadata(
  tensor : NumericTensorSchema,
  dtype : StorageDType,
) -> Bool {
  tensor.storage.dtype() == dtype &&
  tensor.storage.encoding() == StorageEncoding::plain_v1() &&
  !tensor.storage.requires_scale() &&
  !tensor.storage.requires_zero_point_tensor() &&
  !tensor.storage.requires_codebook()
}

///|
fn validate_schema_role(
  tensor : NumericTensorSchema,
) -> Unit raise NumericContractError {
  let valid = if tensor.role == NumericTensorRole::parameter() {
    true
  } else if tensor.role == NumericTensorRole::scale_metadata() {
    schema_plain_metadata(tensor, StorageDType::f32())
  } else if tensor.role == NumericTensorRole::zero_point_metadata() {
    schema_plain_metadata(tensor, StorageDType::i8())
  } else if tensor.role == NumericTensorRole::codebook_metadata() {
    schema_plain_metadata(tensor, StorageDType::f32())
  } else {
    false
  }
  if !valid {
    raise InvalidNumericContract(StorageCombination)
  }
}

///|
fn schema_tensor_at(
  tensors : ArrayView[NumericTensorSchema],
  ordinal : NumericTensorOrdinal,
  counters : NumericSchemaConstructionCounters?,
) -> NumericTensorSchema raise NumericContractError {
  count_metadata_lookup(counters)
  guard tensors.get(ordinal.value) is Some(tensor) && tensor.ordinal == ordinal else {
    raise InvalidNumericContract(Identifier)
  }
  tensor
}

///|
fn claim_metadata_owner(
  owners : FixedArray[Int],
  target : NumericTensorOrdinal,
  source : NumericTensorOrdinal,
  counters : NumericSchemaConstructionCounters?,
) -> Unit raise NumericContractError {
  let target_index = target.value
  if target_index < 0 ||
    target_index >= owners.length() ||
    owners[target_index] != -1 {
    raise InvalidNumericContract(StorageCombination)
  }
  owners[target_index] = source.value
  count_owner_claim(counters)
}

///|
fn validate_schema_scale(
  tensors : ArrayView[NumericTensorSchema],
  source : NumericTensorSchema,
  reference : ScaleTensorOrdinal,
  counters : NumericSchemaConstructionCounters?,
) -> Unit raise NumericContractError {
  let target = schema_tensor_at(tensors, { value: reference.value }, counters)
  if target.role != NumericTensorRole::scale_metadata() ||
    target.ordinal == source.ordinal {
    raise InvalidNumericContract(StorageCombination)
  }
  if source.storage.scale_granularity() == ScaleGranularity::per_tensor() {
    if target.shape.rank() != 0 {
      raise InvalidNumericContract(StorageCombination)
    }
  } else if source.storage.scale_granularity() ==
    ScaleGranularity::per_output_channel() {
    let required_rank = if source.storage.is_symmetric_i8_weight_only_v1() {
      2
    } else {
      source.shape.rank()
    }
    guard source.shape.rank() == required_rank &&
      source.shape.dimension(0) is Some(output_channels) &&
      target.shape.rank() == 1 &&
      target.shape.dimension(0) == Some(output_channels) else {
      raise InvalidNumericContract(StorageCombination)
    }
  } else {
    raise InvalidNumericContract(StorageCombination)
  }
}

///|
fn validate_schema_metadata(
  tensors : ArrayView[NumericTensorSchema],
  source : NumericTensorSchema,
  scale_owners : FixedArray[Int],
  zero_owners : FixedArray[Int],
  codebook_owners : FixedArray[Int],
  counters : NumericSchemaConstructionCounters?,
) -> Unit raise NumericContractError {
  match source.scale {
    Some(reference) => {
      validate_schema_scale(tensors, source, reference, counters)
      claim_metadata_owner(
        scale_owners,
        { value: reference.value },
        source.ordinal,
        counters,
      )
    }
    None => ()
  }
  match source.zero_point {
    Some(reference) => {
      let target = schema_tensor_at(
        tensors,
        { value: reference.value },
        counters,
      )
      guard source.shape.dimension(0) is Some(output_channels) &&
        target.role == NumericTensorRole::zero_point_metadata() &&
        target.ordinal != source.ordinal &&
        target.shape.rank() == 1 &&
        target.shape.dimension(0) == Some(output_channels) else {
        raise InvalidNumericContract(StorageCombination)
      }
      claim_metadata_owner(
        zero_owners,
        { value: reference.value },
        source.ordinal,
        counters,
      )
    }
    None => ()
  }
  match source.codebook {
    Some(reference) => {
      let target = schema_tensor_at(
        tensors,
        { value: reference.value },
        counters,
      )
      if target.role != NumericTensorRole::codebook_metadata() ||
        target.ordinal == source.ordinal ||
        target.shape.rank() != 1 ||
        target.shape.element_count() != 256L {
        raise InvalidNumericContract(StorageCombination)
      }
      claim_metadata_owner(
        codebook_owners,
        { value: reference.value },
        source.ordinal,
        counters,
      )
    }
    None => ()
  }
}

///|
fn new_model_numeric_schema_with_counters(
  limits : ModelNumericSchemaLimits,
  tensors : ArrayView[NumericTensorSchema],
  operations : ArrayView[NumericOperationSchema],
  counters : NumericSchemaConstructionCounters?,
) -> ModelNumericSchema raise NumericContractError {
  let canonical_length = preflight_model_numeric_schema_size(
    limits, tensors, operations, counters,
  )
  for index, tensor in tensors {
    if tensor.ordinal.value != index {
      raise InvalidNumericContract(Identifier)
    }
    validate_schema_role(tensor)
  }
  let scale_owners = FixedArray::make(tensors.length(), -1)
  let zero_owners = FixedArray::make(tensors.length(), -1)
  let codebook_owners = FixedArray::make(tensors.length(), -1)
  count_owner_tables(counters)
  for tensor in tensors {
    validate_schema_metadata(
      tensors, tensor, scale_owners, zero_owners, codebook_owners, counters,
    )
  }
  for index, operation in operations {
    if operation.ordinal.value != index {
      raise InvalidNumericContract(Identifier)
    }
  }
  let owned_tensors = ReadOnlyArray::from_array(tensors)
  let owned_operations = ReadOnlyArray::from_array(operations)
  match counters {
    Some(value) =>
      value.copied_entries += tensors.length() + operations.length()
    None => ()
  }
  let digest = digest_model_numeric_schema(
    canonical_length, owned_tensors, owned_operations,
  )
  match counters {
    Some(value) => {
      value.hash_walks += 1
      if value.preflight_tensors != tensors.length() ||
        value.owner_table_allocations != 3 ||
        value.metadata_lookups != value.owner_claims ||
        value.copied_entries != tensors.length() + operations.length() ||
        value.hash_walks != 1 {
        abort("numeric schema construction counters diverged")
      }
    }
    None => ()
  }
  {
    tensors: owned_tensors,
    operations: owned_operations,
    canonical_length,
    digest,
  }
}
