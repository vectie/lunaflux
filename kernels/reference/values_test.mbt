///|
fn reference_limits() -> @reference.ReferenceLimits raise {
  @reference.ReferenceLimits::new(max_rows=16, max_columns=32, max_elements=256)
}

///|
fn assert_float_close(
  actual : Float,
  expected : Float,
  tolerance : Float,
) -> Unit raise {
  if (actual - expected).abs() > tolerance {
    fail(
      "float mismatch: actual=\{actual.to_double()} expected=\{expected.to_double()}",
    )
  }
}

///|
fn assert_activation_close(
  actual : @reference.Activation,
  rows : Int,
  columns : Int,
  expected : ArrayView[Float],
  tolerance? : Float = 1.0e-5F,
) -> Unit raise {
  assert_eq(actual.rows(), rows)
  assert_eq(actual.columns(), columns)
  assert_eq(actual.values().length(), expected.length())
  for index, value in expected {
    assert_float_close(actual.values()[index], value, tolerance)
  }
}

///|
test "rank-2 values are copied and remain immutable to callers" {
  let source : Array[Float] = [1.0F, 2.0F, 3.0F, 4.0F]
  let activation = @reference.Activation::new(
    rows=2,
    columns=2,
    values=source,
    limits=reference_limits(),
  )
  let tensor = @reference.Tensor::new(
    rows=2,
    columns=2,
    values=source,
    limits=reference_limits(),
  )
  source[0] = 99.0F
  assert_eq(activation.at(row=0, column=0), 1.0F)
  assert_eq(tensor.at(row=1, column=1), 4.0F)
}

///|
test "strict little-endian BF16 decoding preserves signed values" {
  let tensor = @reference.decode_bfloat16_le(
    [0x80, 0x3f, 0x00, 0xc0],
    rows=1,
    columns=2,
    limits=reference_limits(),
  )
  assert_eq(tensor.at(row=0, column=0), 1.0F)
  assert_eq(tensor.at(row=0, column=1), -2.0F)
}
