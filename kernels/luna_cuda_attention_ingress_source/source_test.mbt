///|
test "shared epilogue preserves fold and rounding boundaries across fusion cuts" {
  let epilogue = @luna_projection_tile_compiler.ProjectionAttentionIngressEpilogue::new(
    query_heads=16,
    kv_heads=8,
    head_dimension=128,
    norm_epsilon=0.000001,
    rope_theta=1000000.0,
  )
  let source = lower_attention_ingress_numerics(
    epilogue.numerics(reduction_lanes=32),
  )
  assert_true(source.contains("component < 128; component += 32"))
  assert_true(source.contains("delta = 16; delta > 0; delta >>= 1"))
  assert_true(
    source.contains("__float2bfloat16_rn(__fmul_rn(__fmul_rn(values[component]"),
  )
  assert_true(
    source.contains(
      "sincosf(__fmul_rn((float)position, LF_ROPE_INV_FREQ[pair])",
    ),
  )
  assert_true(
    source.contains(
      "const float first = values[pair], second = values[pair + 64]",
    ),
  )
  assert_false(source.contains("blockDim"))
  assert_false(source.contains("powf("))
  assert_false(source.contains("__syncthreads"))
}
