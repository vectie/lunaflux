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

///|
test "square summation policy preserves BF16 boundaries and ordered default" {
  let ordered = @utf8.decode(
    norm_source(rows=2, width=5120, epsilon=0.000001, symbol="ordered"),
  )
  let compensated = @utf8.decode(
    norm_source(
      rows=2,
      width=5120,
      epsilon=0.000001,
      symbol="accurate",
      square_reduction_law=CompensatedFloat32,
    ),
  )
  assert_true(ordered.contains("sum=__fadd_rn(sum,__fmul_rn(v,v));"))
  assert_false(ordered.contains("correction"))
  assert_true(
    compensated.contains("correction=__fsub_rn(__fsub_rn(next,sum),adjusted);"),
  )
  assert_true(
    compensated.contains("__bfloat162float(__float2bfloat16_rn(__fmul_rn"),
  )
  assert_false(compensated.contains("double"))
  let head = @utf8.decode(
    attention_source(
      rows=2,
      query_heads=4,
      kv_heads=2,
      dimension=128,
      epsilon=0.000001,
      prefix="accurate",
      square_reduction_law=CompensatedFloat32,
    ),
  )
  assert_true(head.contains("float sum=0.0f, correction=0.0f;"))
  assert_true(head.contains("score=__fadd_rn(score,__fmul_rn"))
  assert_true(head.contains("threadIdx.x!=0"))
}

///|
test "export bounded actual normalization lowering for independent GPU oracle" {
  if @env.get_env_var("LUNA_EXPORT_TEXT_NORM") != Some("1") {
    return
  }
  println("LUNA_TEXT_NORM_BEGIN")
  for
    (name, law) in [
      ("ordered", @precision.OrderedFloat32),
      ("compensated", @precision.CompensatedFloat32),
    ] {
    println(
      @utf8.decode(
        norm_source(
          rows=3,
          width=5120,
          epsilon=0.000001,
          symbol=name + "_norm",
          square_reduction_law=law,
        ),
      ),
    )
    println(
      @utf8.decode(
        attention_source(
          rows=3,
          query_heads=4,
          kv_heads=2,
          dimension=128,
          epsilon=0.000001,
          prefix=name,
          square_reduction_law=law,
        ),
      ),
    )
  }
  println("LUNA_TEXT_NORM_END")
}
