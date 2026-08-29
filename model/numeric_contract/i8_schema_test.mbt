///|
fn i8_schema_parameter(
  ordinal : Int,
  shape : ArrayView[Int64],
  scale : Int,
) -> @numeric_contract.NumericTensorSchema {
  try! @numeric_contract.NumericTensorSchema::new(
    ordinal=@numeric_contract.NumericTensorOrdinal::new(ordinal),
    role=@numeric_contract.NumericTensorRole::parameter(),
    shape=@numeric_contract.NumericShape::new(shape),
    storage=@numeric_contract.TensorStorageContract::symmetric_i8_weight_only_v1(),
    scale=@numeric_contract.ScaleTensorOrdinal::new(
      @numeric_contract.NumericTensorOrdinal::new(scale),
    ),
  )
}

///|
fn i8_schema_plain(
  dtype : @numeric_contract.StorageDType,
) -> @numeric_contract.TensorStorageContract {
  try! @numeric_contract.TensorStorageContract::new(
    dtype~,
    layout=@numeric_contract.StorageLayout::dense_row_major_v1(),
    encoding=@numeric_contract.StorageEncoding::plain_v1(),
    scale_granularity=@numeric_contract.ScaleGranularity::absent(),
    zero_point=@numeric_contract.ZeroPointMode::absent(),
    codebook_layout=@numeric_contract.CodebookLayout::absent(),
    conversion=@numeric_contract.NumericConversion::exact_v1(),
  )
}

///|
fn i8_schema_scale(
  ordinal : Int,
  shape : ArrayView[Int64],
  storage? : @numeric_contract.TensorStorageContract = @numeric_contract.TensorStorageContract::symmetric_i8_weight_only_scale_v1(),
) -> @numeric_contract.NumericTensorSchema {
  try! @numeric_contract.NumericTensorSchema::new(
    ordinal=@numeric_contract.NumericTensorOrdinal::new(ordinal),
    role=@numeric_contract.NumericTensorRole::scale_metadata(),
    shape=@numeric_contract.NumericShape::new(shape),
    storage~,
  )
}

///|
fn expect_i8_schema_rejected(
  tensors : ArrayView[@numeric_contract.NumericTensorSchema],
) -> Unit raise {
  try
    @numeric_contract.ModelNumericSchema::new(
      limits=@numeric_contract.ModelNumericSchemaLimits::production(),
      tensors~,
      operations=[],
    )
  catch {
    InvalidNumericContract(StorageCombination) => ()
    InvalidNumericContract(Identifier) => ()
    _ => fail("unexpected symmetric-I8 schema failure")
  } noraise {
    _ => fail("malformed symmetric-I8 schema was admitted")
  }
}

///|
test "symmetric I8 schema binds one exact per-output-row F32 scale" {
  let schema = @numeric_contract.ModelNumericSchema::new(
    limits=@numeric_contract.ModelNumericSchemaLimits::production(),
    tensors=[i8_schema_parameter(0, [4L, 3L], 1), i8_schema_scale(1, [4L])],
    operations=[],
  )
  assert_eq(schema.tensors().length(), 2)
  assert_eq(
    schema.tensors()[0].scale(),
    Some(
      @numeric_contract.ScaleTensorOrdinal::new(
        @numeric_contract.NumericTensorOrdinal::new(1),
      ),
    ),
  )
}

///|
test "symmetric I8 schema rejects scalar wrong-axis and rank-two scales" {
  let bad_shapes : Array[Array[Int64]] = [[], [3L], [4L, 1L]]
  for shape in bad_shapes {
    expect_i8_schema_rejected([
      i8_schema_parameter(0, [4L, 3L], 1),
      i8_schema_scale(1, shape),
    ])
  }
}

///|
test "symmetric I8 schema rejects non-matrix weight rank and wrong scale storage" {
  expect_i8_schema_rejected([
    i8_schema_parameter(0, [4L], 1),
    i8_schema_scale(1, [4L]),
  ])
  expect_i8_schema_rejected([
    i8_schema_parameter(0, [4L, 3L], 1),
    i8_schema_scale(
      1,
      [4L],
      storage=i8_schema_plain(@numeric_contract.StorageDType::bfloat16()),
    ),
  ])
}

///|
test "symmetric I8 schema rejects missing wrong-role and shared scale references" {
  expect_i8_schema_rejected([i8_schema_parameter(0, [4L, 3L], 1)])
  let wrong_role = try! @numeric_contract.NumericTensorSchema::new(
    ordinal=@numeric_contract.NumericTensorOrdinal::new(1),
    role=@numeric_contract.NumericTensorRole::parameter(),
    shape=@numeric_contract.NumericShape::new([4L]),
    storage=i8_schema_plain(@numeric_contract.StorageDType::f32()),
  )
  expect_i8_schema_rejected([i8_schema_parameter(0, [4L, 3L], 1), wrong_role])
  expect_i8_schema_rejected([
    i8_schema_parameter(0, [4L, 3L], 2),
    i8_schema_parameter(1, [4L, 3L], 2),
    i8_schema_scale(2, [4L]),
  ])
}
