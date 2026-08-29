///|
fn schema_test_plain(
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
fn schema_test_bf16_tensor(
  ordinal : Int,
) -> @numeric_contract.NumericTensorSchema {
  try! @numeric_contract.NumericTensorSchema::new(
    ordinal=@numeric_contract.NumericTensorOrdinal::new(ordinal),
    role=@numeric_contract.NumericTensorRole::parameter(),
    shape=@numeric_contract.NumericShape::new([4L, 4L]),
    storage=schema_test_plain(@numeric_contract.StorageDType::bfloat16()),
  )
}

///|
fn schema_test_operation(
  ordinal : Int,
) -> @numeric_contract.NumericOperationSchema {
  let ordinal = try! @numeric_contract.NumericOperationOrdinal::new(ordinal)
  let execution = try! @numeric_contract.OperationExecutionContract::new(
    activation_input=@numeric_contract.ComputeDType::absent(),
    activation_compute=@numeric_contract.ComputeDType::absent(),
    activation_scale=@numeric_contract.ActivationScalePolicy::absent_exact_v1(),
    tensor_input=@numeric_contract.ComputeDType::bfloat16(),
    accumulator=@numeric_contract.AccumulatorDType::f32(),
    output=@numeric_contract.ComputeDType::bfloat16(),
    conversion=@numeric_contract.NumericConversion::exact_v1(),
    order=@numeric_contract.OperationOrderPolicy::strict_declared_order_v1(),
  )
  @numeric_contract.NumericOperationSchema::new(ordinal~, execution~)
}

///|
test "complete schema owns copied tensor and operation tables" {
  let tensors = [schema_test_bf16_tensor(0)]
  let operations = [schema_test_operation(0)]
  let schema = @numeric_contract.ModelNumericSchema::new(
    limits=@numeric_contract.ModelNumericSchemaLimits::production(),
    tensors~,
    operations~,
  )
  let digest = schema.digest()
  tensors.push(schema_test_bf16_tensor(1))
  operations.push(schema_test_operation(1))
  assert_eq(schema.tensors().length(), 1)
  assert_eq(schema.operations().length(), 1)
  assert_eq(schema.digest(), digest)
  assert_eq(schema.canonical_bytes().length(), 233)
  assert_eq(
    schema.digest().as_hex(),
    "7e09d0fe0d3e162c48ce26b562bc8e88846515b43e12433115206401df777694",
  )
  assert_eq(schema.canonical_bytes(), schema.canonical_bytes())
}

///|
test "complete schema rejects noncontiguous stable ordinals" {
  try
    @numeric_contract.ModelNumericSchema::new(
      limits=@numeric_contract.ModelNumericSchemaLimits::production(),
      tensors=[schema_test_bf16_tensor(1)],
      operations=[schema_test_operation(0)],
    )
  catch {
    InvalidNumericContract(Identifier) => ()
    _ => fail("unexpected ordinal failure")
  } noraise {
    _ => fail("noncontiguous tensor ordinal was admitted")
  }
  try
    @numeric_contract.ModelNumericSchema::new(
      limits=@numeric_contract.ModelNumericSchemaLimits::production(),
      tensors=[schema_test_bf16_tensor(0)],
      operations=[schema_test_operation(1)],
    )
  catch {
    InvalidNumericContract(Identifier) => ()
    _ => fail("unexpected operation-ordinal failure")
  } noraise {
    _ => fail("noncontiguous operation ordinal was admitted")
  }
}

///|
test "codebook metadata requires the exact canonical shape" {
  let codebook_storage = try! @numeric_contract.TensorStorageContract::new(
    dtype=@numeric_contract.StorageDType::i8(),
    layout=@numeric_contract.StorageLayout::dense_row_major_v1(),
    encoding=@numeric_contract.StorageEncoding::codebook_v1(),
    scale_granularity=@numeric_contract.ScaleGranularity::absent(),
    zero_point=@numeric_contract.ZeroPointMode::absent(),
    codebook_layout=@numeric_contract.CodebookLayout::contiguous_f32_values_v1(),
    conversion=@numeric_contract.NumericConversion::exact_v1(),
  )
  let parameter = try! @numeric_contract.NumericTensorSchema::new(
    ordinal=@numeric_contract.NumericTensorOrdinal::new(0),
    role=@numeric_contract.NumericTensorRole::parameter(),
    shape=@numeric_contract.NumericShape::new([4L]),
    storage=codebook_storage,
    codebook=@numeric_contract.CodebookTensorOrdinal::new(
      @numeric_contract.NumericTensorOrdinal::new(1),
    ),
  )
  let malformed_codebook = try! @numeric_contract.NumericTensorSchema::new(
    ordinal=@numeric_contract.NumericTensorOrdinal::new(1),
    role=@numeric_contract.NumericTensorRole::codebook_metadata(),
    shape=@numeric_contract.NumericShape::new([16L, 16L]),
    storage=schema_test_plain(@numeric_contract.StorageDType::f32()),
  )
  try
    @numeric_contract.ModelNumericSchema::new(
      limits=@numeric_contract.ModelNumericSchemaLimits::production(),
      tensors=[parameter, malformed_codebook],
      operations=[],
    )
  catch {
    InvalidNumericContract(StorageCombination) => ()
    _ => fail("unexpected codebook-shape failure")
  } noraise {
    _ => fail("noncanonical codebook shape was admitted")
  }
}

///|
test "per-tensor scale metadata requires the scalar shape" {
  let scaled = try! @numeric_contract.TensorStorageContract::new(
    dtype=@numeric_contract.StorageDType::f8_e4m3_finite(),
    layout=@numeric_contract.StorageLayout::dense_row_major_v1(),
    encoding=@numeric_contract.StorageEncoding::scaled_v1(),
    scale_granularity=@numeric_contract.ScaleGranularity::per_tensor(),
    zero_point=@numeric_contract.ZeroPointMode::absent(),
    codebook_layout=@numeric_contract.CodebookLayout::absent(),
    conversion=@numeric_contract.NumericConversion::quantized_v1(),
  )
  let parameter = try! @numeric_contract.NumericTensorSchema::new(
    ordinal=@numeric_contract.NumericTensorOrdinal::new(0),
    role=@numeric_contract.NumericTensorRole::parameter(),
    shape=@numeric_contract.NumericShape::new([4L, 4L]),
    storage=scaled,
    scale=@numeric_contract.ScaleTensorOrdinal::new(
      @numeric_contract.NumericTensorOrdinal::new(1),
    ),
  )
  let rank_one_scale = try! @numeric_contract.NumericTensorSchema::new(
    ordinal=@numeric_contract.NumericTensorOrdinal::new(1),
    role=@numeric_contract.NumericTensorRole::scale_metadata(),
    shape=@numeric_contract.NumericShape::new([1L]),
    storage=schema_test_plain(@numeric_contract.StorageDType::f32()),
  )
  try
    @numeric_contract.ModelNumericSchema::new(
      limits=@numeric_contract.ModelNumericSchemaLimits::production(),
      tensors=[parameter, rank_one_scale],
      operations=[],
    )
  catch {
    InvalidNumericContract(StorageCombination) => ()
    _ => fail("unexpected scale-shape failure")
  } noraise {
    _ => fail("rank-one per-tensor scale was admitted as a scalar")
  }
}
