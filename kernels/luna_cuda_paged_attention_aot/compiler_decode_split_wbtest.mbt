///|
test "portable compiler adapter agrees with runtime row workspace and launch geometry" {
  let shape = @luna_attention_strategy.AttentionShape::new(
    query_heads=16,
    kv_heads=8,
    head_dimension=128,
    tokens_per_page=16,
  )
  let dispatch = @luna_attention_strategy.plan_attention_dispatch(
    shape, 40960, 32, 8,
  )
  let workspace = @luna_attention_strategy.plan_decode_attention_workspace(
    shape,
    dispatch.decode(),
    32,
    40960,
    67108864L,
  )
  let compiled = compile_readonly_decode_split(
    shape,
    @launch_contract.PagedKernelProfile::new(
      id=@launch_contract.KernelProfileId::from_stable_int(9),
      max_query_rows=32,
      max_query_tokens=32,
      max_page_table_entries=2560,
    ),
    @catalog.DeviceTarget::new(
      compute_major=12,
      compute_minor=0,
      supports_bf16=true,
      supports_cublas_lt=false,
    ),
    32768L,
    2560L,
    40960,
    workspace,
    "partial_v2",
    "merge_v2",
    1048576,
  )
  assert_eq(compiled.lowering().key_value_partitions(), 8)
  assert_eq(compiled.source().workspace().total_bytes(), 2129920L)
  assert_true(
    @utf8.decode(compiled.source().bytes()).contains(
      "const __nv_bfloat16 *key_cache",
    ),
  )
}
