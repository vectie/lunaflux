///|
test "ingress CTA row ownership is independent of instruction and epilogue geometry" {
  let baseline = qwen_fused_test_candidate(
    head_dimension=128,
    maximum_tokens=129,
  )
  for tiles in [1, 2, 4] {
    let policy = qwen_ingress_policy(2, 2, 1, 1, rows=tiles)
    let candidate = qwen_fused_test_candidate(
      head_dimension=128,
      maximum_tokens=129,
      ingress_policy=policy,
    )
    let compilation = candidate.tile_compilation()
    assert_eq(
      compilation.program().digest(),
      baseline.tile_compilation().program().digest(),
    )
    assert_eq(
      compilation.schedule().matrix_pipeline().unwrap().row_width(),
      16 * tiles,
    )
    assert_eq(candidate.launch_dimensions().grid_x(), 1 + 128 / (16 * tiles))
    assert_eq(
      candidate.launch_dimensions().grid_y(),
      baseline.launch_dimensions().grid_y(),
    )
    let source = @utf8.decode(candidate.source_bytes())
    assert_true(source.contains("#define LF_TILE_ROWS \{16 * tiles}\n"))
    assert_true(source.contains("float> c[\{tiles}][LF_LIVE_COLUMNS]"))
    assert_true(source.contains("projected[row_fragment*16][column_base]"))
    assert_true(source.contains("const int token = token_begin + local_row"))
    assert_true(source.contains("if (token >= tokens) continue;"))
    assert_true(source.contains("if (tokens == 1)"))
    let production = lower_fused_qwen_qkv_qknorm_rope_paged_kv_write_production_candidate(
      candidate,
      fused_test_limits(),
    )
    assert_true(
      @utf8.decode(production.recipe_bytes()).contains(
        "query_tile_rows=\{16 * tiles}\n",
      ),
    )
    if @env.get_env_var("LUNA_TEST_EXPORT_COUNTER_REPAIR") == Some("1") {
      println("LUNA_SOURCE_BEGIN ingress-\{tiles}")
      println(@utf8.decode(production.source_bytes()))
      println("LUNA_SOURCE_END")
    }
    assert_true(
      production.source_digest() !=
      lower_fused_qwen_qkv_qknorm_rope_paged_kv_write_production_candidate(
        baseline,
        fused_test_limits(),
      ).source_digest(),
    )
  }
  // A wider row map is legal for a smaller complete-head epilogue, but must
  // not bypass the fixed-shared-storage limit for a 128-component head.
  let wide = qwen_ingress_policy(2, 2, 1, 1, rows=8)
  let small = qwen_fused_test_candidate(
    head_dimension=64,
    maximum_tokens=129,
    ingress_policy=wide,
  )
  assert_eq(small.launch_dimensions().grid_x(), 2)
  try
    qwen_fused_test_candidate(
      head_dimension=128,
      maximum_tokens=129,
      ingress_policy=wide,
    )
  catch {
    FusedParallelAotError::FusedParallelUnsupported(StaticResources) => ()
    _ => fail("wrong row tile resource error")
  } noraise {
    _ => fail("oversized row epilogue accepted")
  }
}
