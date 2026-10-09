///|
test "prefix norm renders distinct zero and positive suffix ABIs" {
  let full = prefix_rms_norm_bf16_with_optional_suffix_copy_source(
    prefix_width=512,
    suffix_width=64,
    maximum_rows=32,
    epsilon=0.00001,
    function_symbol="prefix_norm",
  )
  let flash = prefix_rms_norm_bf16_with_optional_suffix_copy_source(
    prefix_width=512,
    suffix_width=0,
    maximum_rows=32,
    epsilon=0.00001,
    function_symbol="prefix_norm",
  )
  let full_text = @utf8.decode(full.source_bytes())
  let flash_text = @utf8.decode(flash.source_bytes())
  assert_true(full_text.contains("__nv_bfloat16 *suffix_output"))
  assert_true(
    full_text.contains("suffix_output[(unsigned long long)row * 64ULL"),
  )
  assert_false(flash_text.contains("suffix_output"))
  assert_true(full_text.contains("column < 512"))
  assert_true(full_text.contains("__fadd_rn"))
  assert_true(full_text.contains("__float2bfloat16_rn"))
}

///|
test "prefix norm rejects invalid suffix epsilon and rows" {
  try
    prefix_rms_norm_bf16_with_optional_suffix_copy_source(
      prefix_width=512,
      suffix_width=-1,
      maximum_rows=32,
      epsilon=0.00001,
      function_symbol="x",
    )
  catch {
    InvalidSuffixWidth => ()
    _ => fail("wrong suffix error")
  } noraise {
    _ => fail("negative suffix admitted")
  }
  try
    prefix_rms_norm_bf16_with_optional_suffix_copy_source(
      prefix_width=512,
      suffix_width=0,
      maximum_rows=0,
      epsilon=0.00001,
      function_symbol="x",
    )
  catch {
    InvalidMaximumRows => ()
    _ => fail("wrong rows error")
  } noraise {
    _ => fail("zero rows admitted")
  }
  try
    prefix_rms_norm_bf16_with_optional_suffix_copy_source(
      prefix_width=512,
      suffix_width=0,
      maximum_rows=32,
      epsilon=0.0,
      function_symbol="x",
    )
  catch {
    InvalidEpsilon => ()
    _ => fail("wrong epsilon error")
  } noraise {
    _ => fail("zero epsilon admitted")
  }
}
