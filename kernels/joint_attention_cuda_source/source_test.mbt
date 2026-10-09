///|
test "staged attention distinguishes packed rotary and refiner scopes" {
  let packed = staged_joint_attention_bf16_source(
    rows=3,
    hidden_size=4,
    head_count=2,
    head_dimension=4,
    rotary_dimension=2,
    epsilon_numerator=1,
    epsilon_denominator=100000,
    apply_rotary=true,
    function_prefix="packed_attention",
  )
  let refiner = staged_joint_attention_bf16_source(
    rows=2,
    hidden_size=4,
    head_count=2,
    head_dimension=4,
    rotary_dimension=0,
    epsilon_numerator=1,
    epsilon_denominator=100000,
    apply_rotary=false,
    function_prefix="refiner_attention",
  )
  assert_true(packed.applies_rotary())
  assert_false(refiner.applies_rotary())
  let packed_source = try! @utf8.decode(packed.source_bytes())
  let refiner_source = try! @utf8.decode(refiner.source_bytes())
  assert_true(packed_source.contains("const float *rotary_cos"))
  assert_false(refiner_source.contains("const float *rotary_cos"))
  assert_true(packed_source.contains("_qkv("))
  assert_true(packed_source.contains("_qk_norm("))
  assert_true(packed_source.contains("_attention("))
  assert_true(packed_source.contains("_output("))
  assert_true(packed_source.contains("rsqrtf"))
  assert_true(packed_source.contains("expf"))
  assert_true(packed_source.contains("__float2bfloat16_rn"))
  assert_false(packed_source.contains("residual"))
}

///|
test "staged attention rejects hostile geometry epsilon and symbol" {
  for rows in [0, -1] {
    try
      staged_joint_attention_bf16_source(
        rows~,
        hidden_size=4,
        head_count=2,
        head_dimension=4,
        rotary_dimension=2,
        epsilon_numerator=1,
        epsilon_denominator=100000,
        apply_rotary=true,
        function_prefix="ok",
      )
    catch {
      InvalidRowCount => ()
      _ => fail("unexpected row rejection")
    } noraise {
      _ => fail("hostile attention rows admitted")
    }
  }
  try
    staged_joint_attention_bf16_source(
      rows=1,
      hidden_size=4,
      head_count=2,
      head_dimension=4,
      rotary_dimension=0,
      epsilon_numerator=0,
      epsilon_denominator=100000,
      apply_rotary=false,
      function_prefix="ok",
    )
  catch {
    InvalidEpsilon => ()
    _ => fail("unexpected epsilon rejection")
  } noraise {
    _ => fail("zero epsilon admitted")
  }
  try
    staged_joint_attention_bf16_source(
      rows=1,
      hidden_size=4,
      head_count=2,
      head_dimension=4,
      rotary_dimension=2,
      epsilon_numerator=1,
      epsilon_denominator=100000,
      apply_rotary=false,
      function_prefix="9bad",
    )
  catch {
    InvalidWidth => ()
    _ => fail("unexpected rotary-scope rejection")
  } noraise {
    _ => fail("refiner rotary geometry admitted")
  }
  try
    staged_joint_attention_bf16_source(
      rows=1,
      hidden_size=4,
      head_count=2,
      head_dimension=4,
      rotary_dimension=0,
      epsilon_numerator=1,
      epsilon_denominator=100000,
      apply_rotary=false,
      function_prefix="9bad",
    )
  catch {
    InvalidFunctionSymbol => ()
    _ => fail("unexpected attention symbol rejection")
  } noraise {
    _ => fail("leading-digit attention symbol admitted")
  }
  try
    staged_joint_attention_bf16_source(
      rows=1,
      hidden_size=4,
      head_count=2147483647,
      head_dimension=2,
      rotary_dimension=0,
      epsilon_numerator=1,
      epsilon_denominator=100000,
      apply_rotary=false,
      function_prefix="ok",
    )
  catch {
    DimensionOverflow => ()
    _ => fail("unexpected attention overflow rejection")
  } noraise {
    _ => fail("overflowing attention geometry admitted")
  }
}
