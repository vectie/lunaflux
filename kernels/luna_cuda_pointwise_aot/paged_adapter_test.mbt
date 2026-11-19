///|
test "standalone KV adapter retains split-plane padded page addressing" {
  guard @elementwise_physical_ir.KvWriteProgram::new(
      query_heads=16,
      kv_heads=2,
      head_dimension=256,
      page_stride_bytes=16384L,
      element_bytes=2,
      owners=128,
    )
    is Some(program) else {
    fail("KV program")
  }
  let source = @utf8.decode(
    @luna_cuda_pointwise_aot.paged_kv_write_source(
      program,
      symbol="paged_kv",
      tokens_per_page=16,
      physical_page_capacity=256,
    )[:],
  )
  assert_true(source.contains("blockIdx.x * 128ULL"))
  assert_true(source.contains("const int position = positions[token]"))
  assert_true(source.contains("physical_pages[table_index]"))
  assert_true(source.contains("physical_page * 8192ULL"))
  assert_true(
    source.contains(
      "key_cache[cache_index] = rotated_qkv[input_base + 4096ULL + component]",
    ),
  )
  assert_true(
    source.contains(
      "value_cache[cache_index] = rotated_qkv[input_base + 4608ULL + component]",
    ),
  )
  assert_true(source.contains("token * 5120ULL"))
}
