///|
test "dual score split is explicitly bindable while the default stays v1" {
  let shape = @luna_attention_strategy.AttentionShape::new(
    query_heads=16,
    kv_heads=8,
    head_dimension=128,
    tokens_per_page=8,
  )
  let profile = @launch_contract.PagedKernelProfile::new(
    id=@launch_contract.KernelProfileId::from_stable_int(9),
    max_query_rows=32,
    max_query_tokens=2048,
    max_page_table_entries=16384,
  )
  let target = @catalog.DeviceTarget::new(
    compute_major=12,
    compute_minor=1,
    supports_bf16=true,
    supports_cublas_lt=false,
  )
  let default = compile_blockwise_decode_split(
    shape, profile, target, 16384L, 16384L, 4096, 8, 16777216L,
  )
  let dual = compile_blockwise_decode_split(
    shape,
    profile,
    target,
    16384L,
    16384L,
    4096,
    8,
    16777216L,
    candidate_id=454,
  )
  assert_eq(default.lowering().launch(), dual.lowering().launch())
  guard default.lowering().physical_program() is BlockwiseKeyProgram(old, _) else {
    fail("missing default law")
  }
  guard dual.lowering().physical_program() is BlockwiseKeyProgram(new, _) else {
    fail("missing dual law")
  }
  assert_eq(old.numeric_law(), "blockwise-f32-probability-v1")
  assert_eq(new.numeric_law(), "dual-score-blockwise-f32-probability-v2")
  assert_eq(dual.source().workspace().total_bytes(), 2129920L)
  assert_true(
    @utf8.decode(dual.source().bytes()).contains("#define LF_SCORE_OWNERS 2"),
  )
  assert_true(
    @utf8.decode(dual.source().bytes()).contains("partition_active[split]"),
  )
}
