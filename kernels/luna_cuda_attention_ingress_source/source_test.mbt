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
  assert_true(source.contains("float first_values[2], second_values[2]"))
  assert_true(source.contains("delta = 16; delta > 0; delta >>= 1"))
  assert_true(
    source.contains(
      "__float2bfloat16_rn(__fmul_rn(__fmul_rn(first_values[item]",
    ),
  )
  assert_true(
    source.contains(
      "sincosf(__fmul_rn((float)position, LF_ROPE_INV_FREQ[pair])",
    ),
  )
  assert_true(
    source.contains(
      "const float sine = rotary.sine[item], cosine = rotary.cosine[item]",
    ),
  )
  assert_false(source.contains("blockDim"))
  assert_false(source.contains("powf("))
  assert_false(source.contains("__syncthreads"))
  assert_false(source.contains("__syncwarp"))
  assert_true(
    source.contains("static __device__ const float LF_ROPE_INV_FREQ[64]"),
  )
  assert_false(source.contains("__constant__"))
  let prepared = lower_prepared_attention_ingress_numerics(
    epilogue.numerics(reduction_lanes=32),
  )
  assert_true(source.has_prefix(prepared))
  assert_false(prepared.contains("void lf_ingress_norm_rotary("))
  assert_true(prepared.contains("void lf_ingress_norm_rotary_prepared("))
}

///|
test "panic CUDA ingress rejects unsupported logical reduction width" {
  let numerics = @luna_projection_tile_compiler.ProjectionAttentionIngressEpilogue::new(
    query_heads=1,
    kv_heads=1,
    head_dimension=128,
    norm_epsilon=0.000001,
    rope_theta=10000.0,
  ).numerics(reduction_lanes=2048)
  ignore(lower_attention_ingress_numerics(numerics))
}
