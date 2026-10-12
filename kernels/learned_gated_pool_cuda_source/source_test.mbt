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
  assert_eq([..source.execution_order()], [0, 1, 2, 3, 4])
}

///|
test "explicit projection alternative reaches source without rewriting other arithmetic" {
  let before = fixture(4, true, "pool4")
  let after = @learned_gated_pool_cuda_source.learned_gated_pool_source(
    before.plan().with_projection_multiply_add(FusedMultiplyAdd),
    symbol_prefix="pool4",
  )
  let text = @utf8.decode(after.source_bytes())
  let original = @utf8.decode(before.source_bytes())
  assert_true(
    text.contains("value = __fmaf_rn(x, __bfloat162float(wkv[offset]), value)"),
  )
  assert_eq(
    text,
    original
    .replace(
      old="__fadd_rn(value, __fmul_rn(x, __bfloat162float(wkv[offset])))",
      new="__fmaf_rn(x, __bfloat162float(wkv[offset]), value)",
    )
    .replace(
      old="__fadd_rn(score, __fmul_rn(x, __bfloat162float(wgate[offset])))",
      new="__fmaf_rn(x, __bfloat162float(wgate[offset]), score)",
    ),
  )
  assert_eq([..after.execution_order()], [..before.execution_order()])
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

///|
async test "offline numeric program fixture exporter" {
  if @env.get_env_var("LUNA_GATED_POOL_FIXTURE_SOURCE_OUTPUT") is Some(path) {
    let fused = @env.get_env_var("LUNA_GATED_POOL_FIXTURE_FMA") == Some("1")
    let parts = [fixture(4, true, "pool4"), fixture(128, false, "pool128")].map(source => {
        if fused {
          @learned_gated_pool_cuda_source.learned_gated_pool_source(
            source.plan().with_projection_multiply_add(FusedMultiplyAdd),
            symbol_prefix=if source.plan().geometry().2 == 4 {
              "pool4"
            } else {
              "pool128"
            },
          ).source_bytes()
        } else {
          source.source_bytes()
        }
      },
    )
    @fs.write_file(path, parts[0] + b"\n" + parts[1], create_mode=CreateNew)
  }
}
