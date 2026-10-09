///|
test "packed documents bound both softmax passes without changing tensor extents" {
  let kernel = staged_joint_attention_bf16_source(
    rows=64,
    hidden_size=4,
    head_count=1,
    head_dimension=4,
    rotary_dimension=0,
    epsilon_numerator=1,
    epsilon_denominator=100000,
    apply_rotary=false,
    function_prefix="segmented",
    document_boundaries=[0, 16, 64],
  )
  let source = @utf8.decode(kernel.source_bytes())
  assert_true(source.contains("int key_begin = 0; int key_end = 16;"))
  assert_true(
    source.contains("if (qr >= 16ULL) { key_begin = 16; key_end = 64; }"),
  )
  assert_true(
    source.contains(
      "float maximum = -INFINITY; for (int kr = key_begin; kr < key_end;",
    ),
  )
  assert_true(
    source.contains(
      "float numerator = 0.0f; for (int kr = key_begin; kr < key_end;",
    ),
  )
  assert_false(source.contains("for (int kr = 0; kr < 64;"))
}

///|
test "invalid or empty packed documents cannot generate attention" {
  for boundaries in [[0], [1, 64], [0, 63], [0, 16, 16, 64], [0, 65, 64]] {
    try
      staged_joint_attention_bf16_source(
        rows=64,
        hidden_size=4,
        head_count=1,
        head_dimension=4,
        rotary_dimension=0,
        epsilon_numerator=1,
        epsilon_denominator=100000,
        apply_rotary=false,
        function_prefix="segmented",
        document_boundaries=boundaries,
      )
    catch {
      InvalidDocumentBoundaries => ()
      _ => fail("wrong boundary rejection")
    } noraise {
      _ => fail("invalid documents admitted")
    }
  }
}
