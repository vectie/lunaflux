///|
test "decode column transfer follows schedule width without numeric conversion" {
  for pair in [(4, "unsigned int"), (8, "uint2"), (16, "uint4")] {
    let (width, packed) = pair
    let source = emit_decode_key_value_transfer(source_test_abi(), width)
    assert_true(source.contains("Contiguous column map: \{width / 2} BF16"))
    assert_true(
      source.contains(
        "reinterpret_cast<const \{packed} *>(key_cache + cache_index)",
      ),
    )
    assert_true(
      source.contains("reinterpret_cast<\{packed} *>(value_tile + linear)"),
    )
    assert_true(source.contains("if (key < active_keys)"))
    assert_true(source.contains("atomicExch(tile_valid, 0)"))
    assert_false(source.contains("__float2bfloat16"))
  }
}

///|
test "unaligned page stride cannot produce vector loads" {
  let abi = CudaPagedAttentionTileAbi::new(
    query_heads=16,
    key_value_heads=4,
    head_dimension=64,
    tokens_per_page=8,
    input_row_width=1536,
    page_stride_values=2049L,
    physical_page_capacity=32L,
    maximum_query_rows=8,
    maximum_query_tokens=8,
    maximum_page_entries=32,
  )
  for width in [4, 8, 16] {
    let source = emit_decode_key_value_transfer(abi, width)
    assert_true(source.contains("Contiguous column map: 1 BF16"))
    assert_false(source.contains("uint2"))
    assert_false(source.contains("uint4"))
  }
}

///|
test "grouped split compiler realizes its vector map deterministically" {
  let lowering = source_test_decode_lowering(candidate_id=430)
  let first = emit_cuda_attention_tile_source(
    lowering,
    source_test_abi(),
    "vector_decode",
    1048576,
  )
  let again = emit_cuda_attention_tile_source(
    lowering,
    source_test_abi(),
    "vector_decode",
    1048576,
  )
  assert_eq(first.bytes(), again.bytes())
  let source = @utf8.decode(first.bytes())
  assert_true(source.contains("Contiguous column map: 4 BF16"))
  assert_true(source.contains("uint2 staged_key = {}"))
  assert_true(
    source.contains(
      "for (int key = split; key < active_keys; key += LF_SPLITS_PER_HEAD)",
    ),
  )
}
