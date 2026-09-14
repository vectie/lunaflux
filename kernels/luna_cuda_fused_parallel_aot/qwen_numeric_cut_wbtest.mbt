///|
test "full ingress diagnostic fixture shares projection problem extents" {
  let export_sources = @env.get_env_var("LUNA_TEST_EXPORT_INGRESS_CUT") ==
    Some("1")
  let page_tokens = if @env.get_env_var("LUNA_TEST_INGRESS_PAGE_TOKENS") ==
    Some("8") {
    8
  } else {
    16
  }
  let candidate = qwen_fused_test_candidate(
    head_dimension=128,
    maximum_tokens=2048,
    query_heads=16,
    kv_heads=8,
    hidden_width=1024,
    page_tokens~,
    export_projection=export_sources,
  )
  let production = lower_fused_qwen_qkv_qknorm_rope_paged_kv_write_production_candidate(
    candidate,
    fused_test_limits(),
  )
  assert_eq(production.semantic_plan().projection_shape(), (1024, 4096))
  assert_eq(production.semantic_plan().attention_shape(), (16, 8, 128))
  if export_sources {
    println("LUNA_SOURCE_BEGIN ingress-cut")
    println(@utf8.decode(production.source_bytes()))
    println("LUNA_SOURCE_END")
  }
}

///|
test "ingress numerical fixtures derive page capacity from page geometry" {
  for page_tokens in [8, 16] {
    let candidate = qwen_fused_test_candidate(
      head_dimension=128,
      maximum_tokens=2048,
      query_heads=16,
      kv_heads=8,
      hidden_width=1024,
      page_tokens~,
    )
    let source = @utf8.decode(candidate.source_bytes())
    assert_true(source.contains("#define LF_TOKENS_PER_PAGE \{page_tokens}\n"))
    assert_true(
      source.contains(
        "#define LF_TOTAL_PAGE_COUNT \{4 * (2048 / page_tokens)}\n",
      ),
    )
  }
}
