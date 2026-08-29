///|
fn scaling_counters() -> NumericSchemaConstructionCounters {
  new_numeric_schema_construction_counters()
}

///|
fn scaling_plain(dtype : StorageDType) -> TensorStorageContract {
  try! TensorStorageContract::new(
    dtype~,
    layout=StorageLayout::dense_row_major_v1(),
    encoding=StorageEncoding::plain_v1(),
    scale_granularity=ScaleGranularity::absent(),
    zero_point=ZeroPointMode::absent(),
    codebook_layout=CodebookLayout::absent(),
    conversion=NumericConversion::exact_v1(),
  )
}

///|
fn scaling_tensor(
  ordinal : Int,
  role : NumericTensorRole,
  shape : NumericShape,
  storage : TensorStorageContract,
  scale? : ScaleTensorOrdinal,
  zero_point? : ZeroPointTensorOrdinal,
  codebook? : CodebookTensorOrdinal,
) -> NumericTensorSchema {
  try! NumericTensorSchema::new(
    ordinal=NumericTensorOrdinal::new(ordinal),
    role~,
    shape~,
    storage~,
    scale?,
    zero_point?,
    codebook?,
  )
}

///|
test "maximum tensor table remains one bounded linear construction" {
  let shape = try! NumericShape::new([])
  let storage = scaling_plain(StorageDType::bfloat16())
  let tensors : Array[NumericTensorSchema] = []
  for ordinal in 0..<MODEL_NUMERIC_SCHEMA_HARD_MAX_TENSORS {
    tensors.push(
      scaling_tensor(ordinal, NumericTensorRole::parameter(), shape, storage),
    )
  }
  let counters = scaling_counters()
  let schema = try! new_model_numeric_schema_with_counters(
    ModelNumericSchemaLimits::production(),
    tensors,
    [],
    Some(counters),
  )
  assert_eq(schema.tensors().length(), MODEL_NUMERIC_SCHEMA_HARD_MAX_TENSORS)
  assert_eq(
    schema.canonical_length,
    49 + 101 * MODEL_NUMERIC_SCHEMA_HARD_MAX_TENSORS,
  )
  assert_eq(counters.preflight_tensors, MODEL_NUMERIC_SCHEMA_HARD_MAX_TENSORS)
  assert_eq(counters.metadata_lookups, 0)
  assert_eq(counters.owner_claims, 0)
  assert_eq(counters.copied_entries, MODEL_NUMERIC_SCHEMA_HARD_MAX_TENSORS)
}

///|
test "ordinal owner tables perform exactly one lookup and claim per scale" {
  let source_count = 2048
  let source_shape = try! NumericShape::new([1L])
  let metadata_shape = try! NumericShape::new([])
  let storage = try! TensorStorageContract::new(
    dtype=StorageDType::f8_e4m3_finite(),
    layout=StorageLayout::dense_row_major_v1(),
    encoding=StorageEncoding::scaled_v1(),
    scale_granularity=ScaleGranularity::per_tensor(),
    zero_point=ZeroPointMode::absent(),
    codebook_layout=CodebookLayout::absent(),
    conversion=NumericConversion::quantized_v1(),
  )
  let tensors : Array[NumericTensorSchema] = []
  for ordinal in 0..<source_count {
    tensors.push(
      scaling_tensor(
        ordinal,
        NumericTensorRole::parameter(),
        source_shape,
        storage,
        scale=ScaleTensorOrdinal::new(
          NumericTensorOrdinal::new(source_count + ordinal),
        ),
      ),
    )
  }
  let metadata_storage = scaling_plain(StorageDType::f32())
  for ordinal in 0..<source_count {
    tensors.push(
      scaling_tensor(
        source_count + ordinal,
        NumericTensorRole::scale_metadata(),
        metadata_shape,
        metadata_storage,
      ),
    )
  }
  let counters = scaling_counters()
  ignore(
    try! new_model_numeric_schema_with_counters(
      ModelNumericSchemaLimits::production(),
      tensors,
      [],
      Some(counters),
    ),
  )
  assert_eq(counters.metadata_lookups, source_count)
  assert_eq(counters.owner_claims, source_count)
  assert_eq(counters.preflight_tensors, source_count * 2)
}

///|
fn assert_duplicate_owner_rejected(
  tensors : Array[NumericTensorSchema],
) -> Unit raise {
  try
    ModelNumericSchema::new(
      limits=ModelNumericSchemaLimits::production(),
      tensors~,
      operations=[],
    )
  catch {
    InvalidNumericContract(StorageCombination) => ()
    _ => fail("unexpected duplicate-owner failure")
  } noraise {
    _ => fail("duplicate metadata owner was admitted")
  }
}

///|
test "scale zero and codebook metadata each have one exact owner" {
  let scale_storage = try! TensorStorageContract::new(
    dtype=StorageDType::f8_e4m3_finite(),
    layout=StorageLayout::dense_row_major_v1(),
    encoding=StorageEncoding::scaled_v1(),
    scale_granularity=ScaleGranularity::per_tensor(),
    zero_point=ZeroPointMode::absent(),
    codebook_layout=CodebookLayout::absent(),
    conversion=NumericConversion::quantized_v1(),
  )
  let scalar = try! NumericShape::new([])
  let vector = try! NumericShape::new([4L])
  assert_duplicate_owner_rejected([
    scaling_tensor(
      0,
      NumericTensorRole::parameter(),
      vector,
      scale_storage,
      scale=ScaleTensorOrdinal::new(NumericTensorOrdinal::new(2)),
    ),
    scaling_tensor(
      1,
      NumericTensorRole::parameter(),
      vector,
      scale_storage,
      scale=ScaleTensorOrdinal::new(NumericTensorOrdinal::new(2)),
    ),
    scaling_tensor(
      2,
      NumericTensorRole::scale_metadata(),
      scalar,
      scaling_plain(StorageDType::f32()),
    ),
  ])
  let affine = try! TensorStorageContract::new(
    dtype=StorageDType::i8(),
    layout=StorageLayout::dense_row_major_v1(),
    encoding=StorageEncoding::affine_v1(),
    scale_granularity=ScaleGranularity::per_output_channel(),
    zero_point=ZeroPointMode::per_output_channel_tensor_v1(),
    codebook_layout=CodebookLayout::absent(),
    conversion=NumericConversion::quantized_v1(),
  )
  assert_duplicate_owner_rejected([
    scaling_tensor(
      0,
      NumericTensorRole::parameter(),
      vector,
      affine,
      scale=ScaleTensorOrdinal::new(NumericTensorOrdinal::new(3)),
      zero_point=ZeroPointTensorOrdinal::new(NumericTensorOrdinal::new(2)),
    ),
    scaling_tensor(
      1,
      NumericTensorRole::parameter(),
      vector,
      affine,
      scale=ScaleTensorOrdinal::new(NumericTensorOrdinal::new(4)),
      zero_point=ZeroPointTensorOrdinal::new(NumericTensorOrdinal::new(2)),
    ),
    scaling_tensor(
      2,
      NumericTensorRole::zero_point_metadata(),
      vector,
      scaling_plain(StorageDType::i8()),
    ),
    scaling_tensor(
      3,
      NumericTensorRole::scale_metadata(),
      vector,
      scaling_plain(StorageDType::f32()),
    ),
    scaling_tensor(
      4,
      NumericTensorRole::scale_metadata(),
      vector,
      scaling_plain(StorageDType::f32()),
    ),
  ])
  let codebook = try! TensorStorageContract::new(
    dtype=StorageDType::i8(),
    layout=StorageLayout::dense_row_major_v1(),
    encoding=StorageEncoding::codebook_v1(),
    scale_granularity=ScaleGranularity::absent(),
    zero_point=ZeroPointMode::absent(),
    codebook_layout=CodebookLayout::contiguous_f32_values_v1(),
    conversion=NumericConversion::exact_v1(),
  )
  let book = try! NumericShape::new([256L])
  assert_duplicate_owner_rejected([
    scaling_tensor(
      0,
      NumericTensorRole::parameter(),
      vector,
      codebook,
      codebook=CodebookTensorOrdinal::new(NumericTensorOrdinal::new(2)),
    ),
    scaling_tensor(
      1,
      NumericTensorRole::parameter(),
      vector,
      codebook,
      codebook=CodebookTensorOrdinal::new(NumericTensorOrdinal::new(2)),
    ),
    scaling_tensor(
      2,
      NumericTensorRole::codebook_metadata(),
      book,
      scaling_plain(StorageDType::f32()),
    ),
  ])
}

///|
test "canonical overflow rejects before owner tables copies and hash" {
  let tensors = [
    scaling_tensor(
      0,
      NumericTensorRole::parameter(),
      try! NumericShape::new([]),
      scaling_plain(StorageDType::bfloat16()),
    ),
  ]
  let limits = try! ModelNumericSchemaLimits::new(
    max_tensors=1,
    max_operations=0,
    max_rank=0,
    max_canonical_bytes=149,
  )
  let counters = scaling_counters()
  try
    new_model_numeric_schema_with_counters(limits, tensors, [], Some(counters))
  catch {
    NumericSchemaLimitExceeded(SchemaCanonicalBytes) => ()
    _ => fail("unexpected canonical preflight failure")
  } noraise {
    _ => fail("oversize canonical schema was admitted")
  }
  assert_eq(counters.preflight_tensors, 1)
  assert_eq(counters.owner_table_allocations, 0)
  assert_eq(counters.copied_entries, 0)
  assert_eq(counters.hash_walks, 0)
}

///|
fn assert_zero_padded_le(
  bytes : Bytes,
  offset : Int,
  value : Byte,
) -> Unit raise {
  assert_eq(bytes[offset], value)
  for index in 1..<8 {
    assert_eq(bytes[offset + index], 0x00)
  }
}

///|
test "manual schema offsets match the independently derived size formula" {
  let schema = try! ModelNumericSchema::new(
    limits=ModelNumericSchemaLimits::production(),
    tensors=[
      scaling_tensor(
        0,
        NumericTensorRole::parameter(),
        NumericShape::new([4L, 4L]),
        scaling_plain(StorageDType::bfloat16()),
      ),
    ],
    operations=[],
  )
  let bytes = schema.canonical_bytes()
  assert_eq(bytes.length(), 166)
  assert_eq(bytes[0:33], b"lunaflux.model-numeric-schema.v2\x00")
  assert_zero_padded_le(bytes, 33, 1)
  assert_zero_padded_le(bytes, 41, 0)
  assert_zero_padded_le(bytes, 49, 1)
  assert_zero_padded_le(bytes, 57, 2)
  assert_zero_padded_le(bytes, 65, 4)
  assert_zero_padded_le(bytes, 73, 4)
  assert_zero_padded_le(bytes, 81, 45)
  assert_eq(bytes[89:125], b"lunaflux.tensor-storage-contract.v1\x00")
  assert_zero_padded_le(bytes, 134, 0)
  assert_zero_padded_le(bytes, 142, 0)
  assert_zero_padded_le(bytes, 150, 0)
  assert_zero_padded_le(bytes, 158, 0)
}
