///|
test "declared fused qualification shape set covers exact page boundaries" {
  let shapes = declared_qualification_shapes()
  assert_eq(shapes.length(), 4)
  for ordinal, shape in shapes {
    assert_eq(shape.ordinal(), ordinal)
    assert_true(shape.token_count() >= 1 && shape.token_count() <= 8)
    assert_eq(shape.positions().length(), shape.token_count())
    assert_eq(shape.step_counts()[2], 1)
    assert_eq(shape.step_counts()[3], shape.token_count())
    assert_eq(shape.query_row_offsets()[1], shape.token_count())
    assert_eq(shape.page_table_offsets()[1], shape.page_table().length())
    assert_eq(
      shape.sequence_lengths()[0],
      shape.positions()[shape.token_count() - 1] + 1,
    )
    assert_eq(
      shape.expected_dispatch_canary(7U),
      7U * shape.token_count().reinterpret_as_uint(),
    )
    for position in shape.positions() {
      assert_true(position >= 0 && position < 32)
    }
  }
  assert_eq(shapes[1].positions()[0], 15)
  assert_eq(shapes[2].positions()[0], 15)
  assert_eq(shapes[2].positions()[1], 16)
  assert_eq(shapes[3].token_count(), 8)
}

///|
test "independent fused scalar referees cover every declared shape" {
  for shape in declared_qualification_shapes() {
    let qkv = qkv_qualification_fixture(shape)
    assert_eq(
      qkv.expected_rotated_qkv().length(),
      shape.token_count() * qualification_output_width(),
    )
    assert_eq(qkv.expected_key_cache().length(), qualification_cache_elements())
    assert_eq(
      qkv.expected_value_cache().length(),
      qualification_cache_elements(),
    )
    assert_eq(
      try! qualify_qkv_bf16_output(
        qkv.expected_rotated_qkv(),
        qkv.expected_rotated_qkv(),
      ),
      0.0F,
    )
    let qkv_summary = try! compare_qkv_bf16_output(
      qkv.expected_rotated_qkv(),
      qkv.expected_rotated_qkv(),
    )
    assert_eq(qkv_summary.maximum_absolute_error_ppb(), 0L)
    assert_eq(qkv_summary.maximum_relative_error_ppb(), 0L)
    let residual = residual_norm_qualification_fixture(shape)
    assert_eq(
      residual.expected_residual_output().length(),
      shape.token_count() * qualification_input_width(),
    )
    assert_eq(
      try! qualify_residual_norm_bf16_output(
        residual.expected_norm_output(),
        residual.expected_norm_output(),
      ),
      0.0F,
    )
    let residual_summary = try! compare_residual_norm_bf16_output(
      residual.expected_norm_output(),
      residual.expected_norm_output(),
    )
    assert_eq(residual_summary.maximum_absolute_error_ppb(), 0L)
    assert_eq(residual_summary.maximum_relative_error_ppb(), 0L)
  }
}

///|
test "hostile referee output corruption fails at its exact cell" {
  let fixture = qkv_qualification_fixture(declared_qualification_shapes()[2])
  let corrupted = fixture.expected_rotated_qkv()[:].to_owned()
  corrupted[129] = 0x4280
  try qualify_qkv_bf16_output(corrupted, fixture.expected_rotated_qkv()) catch {
    RefereeMismatch(index=129) => ()
    _ => fail("unexpected referee corruption rejection")
  } noraise {
    _ => fail("corrupted fused output passed the scalar referee")
  }
}

///|
test "typed referee rejects non-finite and unbounded intermediate errors" {
  for
    pair in [
      ([0x7f80], [0x3f80]),
      ([0x7fc0], [0x3f80]),
      ([0x7f7f], [0xff7f]),
      ([0x0002], [0x0001]),
      ([0x0001], [0x0000]),
    ] {
    let (actual, expected) = pair
    try compare_qkv_bf16_output(actual, expected) catch {
      RefereeMismatch(index=0) => ()
      _ => fail("unexpected hostile numerical rejection")
    } noraise {
      _ => fail("hostile numerical observation passed")
    }
  }
  let exact_zero = compare_qkv_bf16_output([0x0000], [0x0000])
  assert_eq(exact_zero.maximum_absolute_error_ppb(), 0L)
  assert_eq(exact_zero.maximum_relative_error_ppb(), 0L)
}
