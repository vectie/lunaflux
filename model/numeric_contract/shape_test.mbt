///|
test "shape owner copies dimensions and derives exact storage bytes" {
  let source = [2L, 3L, 4L]
  let shape = @numeric_contract.NumericShape::new(source)
  source[0] = 9L
  assert_eq(shape.rank(), 3)
  assert_eq(shape.dimension(0), Some(2L))
  assert_eq(shape.dimension(2), Some(4L))
  assert_eq(shape.dimension(3), None)
  assert_eq(shape.element_count(), 24L)
  assert_eq(
    @numeric_contract.storage_byte_length(
      @numeric_contract.StorageDType::bfloat16(),
      shape,
    ),
    48L,
  )
  assert_eq(
    @numeric_contract.storage_byte_length(
      @numeric_contract.StorageDType::f8_e4m3_finite(),
      shape,
    ),
    24L,
  )
  assert_eq(
    @numeric_contract.storage_byte_length(
      @numeric_contract.StorageDType::i8(),
      shape,
    ),
    24L,
  )
  assert_eq(
    @numeric_contract.storage_byte_length(
      @numeric_contract.StorageDType::f32(),
      shape,
    ),
    96L,
  )
}

///|
test "shape rejects rank above the global hard cap before owning dimensions" {
  let dimensions = Array::make(17, 1L)
  try @numeric_contract.NumericShape::new(dimensions) catch {
    NumericSchemaLimitExceeded(SchemaRank) => ()
    _ => fail("unexpected rank-limit failure")
  } noraise {
    _ => fail("rank above sixteen was admitted")
  }
  dimensions[0] = 2L
  assert_eq(dimensions.length(), 17)
}

///|
test "scalar shape has one element" {
  let scalar = @numeric_contract.NumericShape::new([])
  assert_eq(scalar.rank(), 0)
  assert_eq(scalar.element_count(), 1L)
  assert_eq(
    @numeric_contract.storage_byte_length(
      @numeric_contract.StorageDType::f32(),
      scalar,
    ),
    4L,
  )
}

///|
test "shape and byte arithmetic reject invalid or overflowing products" {
  for dimensions in [[0L], [-1L], [2L, 0L]] {
    try @numeric_contract.NumericShape::new(dimensions) catch {
      InvalidNumericContract(ShapeDimension) => ()
      _ => fail("unexpected invalid-dimension error")
    } noraise {
      _ => fail("nonpositive shape dimension was admitted")
    }
  }
  try @numeric_contract.NumericShape::new([9223372036854775807L, 2L]) catch {
    InvalidNumericContract(ShapeArithmetic) => ()
    _ => fail("unexpected shape-overflow error")
  } noraise {
    _ => fail("overflowing shape product was admitted")
  }
  let maximum = @numeric_contract.NumericShape::new([9223372036854775807L])
  assert_eq(
    @numeric_contract.storage_byte_length(
      @numeric_contract.StorageDType::i8(),
      maximum,
    ),
    9223372036854775807L,
  )
  try
    @numeric_contract.storage_byte_length(
      @numeric_contract.StorageDType::bfloat16(),
      maximum,
    )
  catch {
    InvalidNumericContract(StorageArithmetic) => ()
    _ => fail("unexpected storage-overflow error")
  } noraise {
    _ => fail("overflowing storage size was admitted")
  }
}
