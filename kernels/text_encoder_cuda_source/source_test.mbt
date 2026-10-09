///|
test "causal GQA source keeps normalized and rotary BF16 stage boundaries" {
  let text = @utf8.decode(
    attention_source(
      rows=3,
      query_heads=4,
      kv_heads=2,
      dimension=8,
      epsilon=0.000001,
      prefix="encoder",
    ),
  )
  assert_true(text.contains("const int kh=h/2;"))
  assert_true(text.contains("key<=row"))
  assert_true(text.contains("__bfloat162float(__float2bfloat16_rn"))
  assert_true(text.contains("d+4"))
  try
    attention_source(
      rows=3,
      query_heads=3,
      kv_heads=2,
      dimension=8,
      epsilon=0.000001,
      prefix="encoder",
    )
  catch {
    InvalidShape => ()
    _ => fail("wrong error")
  } noraise {
    _ => fail("invalid GQA admitted")
  }
}
