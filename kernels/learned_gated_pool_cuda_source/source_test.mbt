///|
fn fixture(
  ratio : Int,
  overlap : Bool,
  prefix : String,
) -> @learned_gated_pool_cuda_source.LearnedGatedPoolCudaSource raise {
  @learned_gated_pool_cuda_source.learned_gated_pool_source(
    @precision.LearnedGatedPoolPrecision::new(
      input_width=8,
      feature_width=16,
      ratio~,
      preceding_window_overlap=overlap,
      maximum_rows=5,
      maximum_positions=1024,
      norm_epsilon=1.0e-6,
      maximum_state_bytes=1048576L,
      maximum_workspace_bytes=1048576L,
    ),
    symbol_prefix=prefix,
  )
}

///|
test "learned gated pooling preserves F32 projections and cross-chunk overlap" {
  let source = fixture(4, true, "pool4")
  let text = @utf8.decode(source.source_bytes())
  assert_true(text.contains("const __nv_bfloat16 *wkv"))
  assert_true(text.contains("float *kv_state, float *score_state"))
  assert_true(text.contains("append[0] + row"))
  assert_true(text.contains("score_state[offset] = -INFINITY"))
  assert_false(text.contains("cuda_fp8"))
  assert_eq(source.launch(0), (1, 1, 1))
  assert_eq(source.launch(1), (5, 1, 256))
  assert_eq(source.launch(3), (2, 1, 256))
  assert_eq(fixture(128, false, "pool128").launch(3), (1, 1, 256))
}

///|
test "offline learned gated pool exporter" {
  if @env.get_env_var("LUNA_EXPORT_GATED_POOL") != Some("1") {
    return
  }
  println("LUNA_GATED_POOL_SOURCE_BEGIN")
  println(@utf8.decode(fixture(4, true, "pool4").source_bytes()))
  println(@utf8.decode(fixture(128, false, "pool128").source_bytes()))
  println("LUNA_GATED_POOL_SOURCE_END")
}
