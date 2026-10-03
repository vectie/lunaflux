///|
test "cached helper reads exact per token pairs without repeating trigonometry" {
  let plan = @luna_projection_tile_compiler.ProjectionAttentionIngressEpilogue::new(
    query_heads=16,
    kv_heads=8,
    head_dimension=128,
    norm_epsilon=0.000001,
    rope_theta=1000000.0,
  ).numerics(reduction_lanes=32)
  let source = lower_cached_attention_ingress_numerics(plan)
  assert_true(
    source.contains(
      "lf_ingress_load_rotary(const float2* cache, int token, int lane)",
    ),
  )
  assert_true(source.contains("cache[(unsigned long long)token * 64 + pair]"))
  assert_false(source.contains("sincosf"))
  assert_false(source.contains("LF_ROPE_INV_FREQ"))
  assert_false(source.contains("lf_ingress_prepare_rotary"))
  assert_false(source.contains("__syncwarp"))
}

///|
test "aligned pair retention retains exact fold order and both BF16 rounds" {
  for dimension in [64, 128] {
    let source = lower_prepared_attention_ingress_numerics(
      @luna_projection_tile_compiler.ProjectionAttentionIngressEpilogue::new(
        query_heads=1,
        kv_heads=1,
        head_dimension=dimension,
        norm_epsilon=0.000001,
        rope_theta=10000.0,
      ).numerics(reduction_lanes=32),
    )
    assert_true(
      source.contains(
        "square_sum = __fadd_rn(square_sum, __fmul_rn(first, first))",
      ),
    )
    assert_true(
      source.contains(
        "square_sum = __fadd_rn(square_sum, __fmul_rn(second, second))",
      ),
    )
    assert_true(
      source.contains(
        "__float2bfloat16_rn(__fmul_rn(__fmul_rn(first_values[item], inverse_rms)",
      ),
    )
    assert_true(
      source.contains("__float2bfloat16_rn(__fsub_rn(__fmul_rn(first, cosine)"),
    )
    assert_false(source.contains("__syncwarp"))
  }
}

///|
test "cross owner split halves retain explicit scratch publications" {
  for dimension in [16, 32, 96] {
    let plan = @luna_projection_tile_compiler.ProjectionAttentionIngressEpilogue::new(
      query_heads=1,
      kv_heads=1,
      head_dimension=dimension,
      norm_epsilon=0.000001,
      rope_theta=10000.0,
    ).numerics(reduction_lanes=32)
    assert_eq(plan.pair_retention(), None)
    let source = lower_prepared_attention_ingress_numerics(plan)
    assert_true(source.contains("__syncwarp"))
    assert_false(source.contains("first_values["))
  }
}
