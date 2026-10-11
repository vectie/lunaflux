///|
test "explicit online BF16 publishes probabilities but not the denominator" {
  for block in [64, 128] {
    let source = @utf8.decode(
      attention_source(
        rows=140,
        query_heads=4,
        kv_heads=2,
        dimension=8,
        epsilon=0.000001,
        prefix="probe",
        probability_law=OnlineBf16(block_tokens=block),
      ),
    )
    assert_true(source.contains("first+=" + block.to_string()))
    assert_true(source.contains("denominator=__fadd_rn(denominator,p);"))
    assert_true(source.contains("__float2bfloat16_rn(p)"))
    assert_true(source.contains("__fmaf_rn(rounded"))
    assert_false(source.contains("key<=row"))
  }
  for
    law in [
      @precision.AttentionProbabilityLaw::GlobalF32,
      OnlineBf16(block_tokens=0),
      OnlineBf16(block_tokens=63),
    ] {
    try
      attention_source(
        rows=3,
        query_heads=4,
        kv_heads=2,
        dimension=8,
        epsilon=0.000001,
        prefix="invalid",
        probability_law=law,
      )
    catch {
      InvalidProbabilityLaw => ()
      _ => fail("wrong probability error")
    } noraise {
      _ => fail("unsupported probability law accepted")
    }
  }
}

///|
test "explicit ordered online and omitted law preserve identical default source" {
  let default = attention_source(
    rows=3,
    query_heads=4,
    kv_heads=2,
    dimension=8,
    epsilon=0.000001,
    prefix="same",
  )
  let explicit = attention_source(
    rows=3,
    query_heads=4,
    kv_heads=2,
    dimension=8,
    epsilon=0.000001,
    prefix="same",
    probability_law=OrderedOnlineF32,
  )
  assert_eq(default, explicit)
}

///|
test "export actual probability kernels for independent physical component test" {
  if @env.get_env_var("LUNA_EXPORT_TEXT_PROBABILITY") != Some("1") {
    return
  }
  println("LUNA_TEXT_PROBABILITY_BEGIN")
  for block in [64, 128] {
    println(
      @utf8.decode(
        attention_source(
          rows=140,
          query_heads=4,
          kv_heads=2,
          dimension=8,
          epsilon=0.000001,
          prefix="probability" + block.to_string(),
          probability_law=OnlineBf16(block_tokens=block),
        ),
      ),
    )
  }
  println("LUNA_TEXT_PROBABILITY_END")
}
