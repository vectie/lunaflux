///|
test "invalid rank element counts and caller bounds fail before allocation" {
  try
    @reference.ReferenceLimits::new(max_rows=0, max_columns=2, max_elements=4)
  catch {
    InvalidLimit(MaxRows) => ()
    _ => fail("unexpected limit error")
  } noraise {
    _ => fail("expected invalid limit")
  }
  let limits = @reference.ReferenceLimits::new(
    max_rows=1,
    max_columns=2,
    max_elements=2,
  )
  try
    @reference.Activation::new(rows=2, columns=1, values=[1.0F, 2.0F], limits~)
  catch {
    LimitExceeded(MaxRows) => ()
    _ => fail("unexpected bounds error")
  } noraise {
    _ => fail("expected bounds failure")
  }
  try @reference.Tensor::new(rows=1, columns=2, values=[1.0F], limits~) catch {
    InvalidShape(ElementCount) => ()
    _ => fail("unexpected element-count error")
  } noraise {
    _ => fail("expected element-count failure")
  }
}

///|
test "non-finite source computed and BF16 values fail closed" {
  let limits = reference_limits()
  let infinity = Float::reinterpret_from_uint(0x7f800000)
  for value in [infinity, -infinity, Float::reinterpret_from_uint(0x7fc00000)] {
    try
      @reference.Activation::new(rows=1, columns=1, values=[value], limits~)
    catch {
      NonFinite(ActivationValue) => ()
      _ => fail("unexpected finite-value error")
    } noraise {
      _ => fail("expected non-finite rejection")
    }
  }
  try
    @reference.decode_bfloat16_le([0xc0, 0x7f], rows=1, columns=1, limits~)
  catch {
    NonFinite(Bfloat16Value) => ()
    _ => fail("unexpected BF16 error")
  } noraise {
    _ => fail("expected BF16 NaN rejection")
  }
  let huge = @reference.Activation::new(
    rows=1,
    columns=2,
    values=[3.4e38F, 3.4e38F],
    limits~,
  )
  let identity = @reference.Tensor::new(
    rows=1,
    columns=2,
    values=[1.0F, 1.0F],
    limits~,
  )
  try @reference.dense_projection(huge, identity, limits~) catch {
    NonFinite(ComputedValue) => ()
    _ => fail("unexpected computed-value error")
  } noraise {
    _ => fail("expected overflow rejection")
  }
}

///|
test "illegal semantic shapes and indexes are typed failures" {
  let limits = reference_limits()
  let input = @reference.Activation::new(
    rows=1,
    columns=2,
    values=[1.0F, 2.0F],
    limits~,
  )
  let wrong_weight = @reference.Tensor::new(
    rows=1,
    columns=1,
    values=[1.0F],
    limits~,
  )
  try @reference.dense_projection(input, wrong_weight, limits~) catch {
    InvalidShape(ProjectionInputWidth) => ()
    _ => fail("unexpected projection-shape error")
  } noraise {
    _ => fail("expected projection shape failure")
  }
  try input.at(row=1, column=0) catch {
    InvalidIndex => ()
    _ => fail("unexpected index error")
  } noraise {
    _ => fail("expected invalid index")
  }
  try
    @reference.decode_bfloat16_le(
      [0x80, 0x3f, 0x00],
      rows=1,
      columns=2,
      limits~,
    )
  catch {
    InvalidShape(Bfloat16ByteCount) => ()
    _ => fail("unexpected BF16 length error")
  } noraise {
    _ => fail("expected BF16 length failure")
  }
}
