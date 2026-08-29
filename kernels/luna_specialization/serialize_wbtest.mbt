///|
fn qk_normalization_shape(
  query_heads : Int,
  kv_heads : Int,
  head_dimension : Int,
  epsilon : Double,
) -> @plan.OperationShape {
  QkNormalizationShape(query_heads~, kv_heads~, head_dimension~, epsilon~)
}

///|
test "Q/K normalization has an appended stable shape tag and exact fields" {
  assert_eq(operation_shape_code(EmbeddingShape(vocabulary_size=8, width=4)), 1)
  assert_eq(
    operation_shape_code(NormalizationShape(width=4, epsilon=0.00001)),
    2,
  )
  assert_eq(
    operation_shape_code(GatedMlpShape(hidden_size=4, intermediate_size=8)),
    7,
  )
  let shape = qk_normalization_shape(2, 1, 4, 0.000001)
  assert_eq(operation_shape_code(shape), 8)
  let first : LunaSpecializationEncoder = { bytes: [], maximum: 64 }
  let second : LunaSpecializationEncoder = { bytes: [], maximum: 64 }
  first.append_operation_shape(shape)
  second.append_operation_shape(shape)
  assert_eq(first.bytes, second.bytes)
  assert_eq(first.bytes.length(), 40)
  assert_eq(first.bytes[0], 8)
  assert_eq(first.bytes[8], 2)
  assert_eq(first.bytes[16], 1)
  assert_eq(first.bytes[24], 4)
}

///|
test "Q/K normalization canonical bytes bind every dimension and epsilon" {
  let encode = fn(shape : @plan.OperationShape) -> Array[Byte] raise {
    let encoder : LunaSpecializationEncoder = { bytes: [], maximum: 64 }
    encoder.append_operation_shape(shape)
    encoder.bytes
  }
  let baseline = encode(qk_normalization_shape(2, 1, 4, 0.000001))
  for
    changed in [
      qk_normalization_shape(4, 1, 4, 0.000001),
      qk_normalization_shape(2, 2, 4, 0.000001),
      qk_normalization_shape(2, 1, 8, 0.000001),
      qk_normalization_shape(2, 1, 4, 0.00001),
    ] {
    assert_true(encode(changed) != baseline)
  }
}
