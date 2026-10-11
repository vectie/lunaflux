///|
test "global BF16 probability law preserves the F32 denominator and explicit contract" {
  for
    law in [
      @precision.AttentionProbabilityLaw::GlobalF32,
      GlobalUnnormalizedBf16,
    ] {
    let rendered = staged_joint_attention_bf16_source(
      rows=3,
      hidden_size=5376,
      head_count=56,
      head_dimension=128,
      rotary_dimension=0,
      epsilon_numerator=1,
      epsilon_denominator=100000,
      apply_rotary=false,
      function_prefix="refiner_probability",
      probability_law=law,
    )
    let source = @utf8.decode(rendered.source_bytes())
    assert_true(
      source.contains("denominator = __fadd_rn(denominator, probability)"),
    )
    if law is GlobalUnnormalizedBf16 {
      assert_true(
        source.contains(
          "__fmul_rn(__bfloat162float(__float2bfloat16_rn(probability))",
        ),
      )
      assert_eq(
        rendered.numerical_contract(),
        GlobalBf16ProbabilityF32DenominatorStagedAttentionV1,
      )
    } else {
      assert_true(source.contains("__fmul_rn(probability,"))
      assert_eq(
        rendered.numerical_contract(),
        OrderedF32AccumulationBf16StagedAttentionV1,
      )
    }
  }
  if @env.get_env_var("LUNA_EXPORT_REFINER_PROBABILITY") is Some(_) {
    println("LUNA_REFINER_PROBABILITY_BEGIN")
    for
      (name, law) in [
        ("refiner_f32", @precision.AttentionProbabilityLaw::GlobalF32),
        ("refiner_bf16", GlobalUnnormalizedBf16),
      ] {
      let source = staged_joint_attention_bf16_source(
        rows=3,
        hidden_size=5376,
        head_count=56,
        head_dimension=128,
        rotary_dimension=0,
        epsilon_numerator=1,
        epsilon_denominator=100000,
        apply_rotary=false,
        function_prefix=name,
        probability_law=law,
      )
      println(@utf8.decode(source.source_bytes()))
    }
    println("LUNA_REFINER_PROBABILITY_END")
  }
}

///|
test "online probability partition is explicit and preserves unrounded mass" {
  let rendered = staged_joint_attention_bf16_source(
    rows=3,
    hidden_size=4,
    head_count=2,
    head_dimension=4,
    rotary_dimension=0,
    epsilon_numerator=1,
    epsilon_denominator=100000,
    apply_rotary=false,
    function_prefix="online",
    probability_law=OnlineBf16(block_tokens=64),
    document_boundaries=[0, 2, 3],
  )
  let source = @utf8.decode(rendered.source_bytes())
  assert_eq(
    rendered.numerical_contract(),
    OnlineBf16ProbabilityF32DenominatorStagedAttentionV1(block_tokens=64),
  )
  assert_true(source.contains("start = end"))
  assert_true(source.contains("min(64, key_end - start)"))
  assert_true(
    source.contains(
      "tile_denominator = __fadd_rn(tile_denominator, probability)",
    ),
  )
  assert_true(source.contains("numerator = __fmul_rn(numerator, alpha)"))
  assert_true(source.contains("__float2bfloat16_rn(probability)"))
}

///|
test "online renderer rejects an invalid probability partition" {
  try
    staged_joint_attention_bf16_source(
      rows=3,
      hidden_size=4,
      head_count=2,
      head_dimension=4,
      rotary_dimension=0,
      epsilon_numerator=1,
      epsilon_denominator=100000,
      apply_rotary=false,
      function_prefix="invalid_online",
      probability_law=OnlineBf16(block_tokens=63),
    )
  catch {
    InvalidProbabilityLaw => ()
    error => raise error
  } noraise {
    _ => fail("different numerical partition was silently substituted")
  }
}
