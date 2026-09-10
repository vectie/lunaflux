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
  let sections = source.split("for (int key_base =") |> Iter::collect
  assert_eq(sections.length(), 2)
  assert_true(sections[0].contains("float query_fragment["))
  assert_true(
    sections[0].contains("rotated_qkv_rows[query_offset + component]"),
  )
  assert_false(sections[1].contains("rotated_qkv_rows["))
  assert_true(
    source.contains(
      "for (int key = split; key < active_keys; key += LF_SPLITS_PER_HEAD)",
    ),
  )
}

///|
test "two-stage decode keeps ordered fold and explicit producer consumer lifetimes" {
  let lowering = source_test_decode_lowering(candidate_id=440)
  assert_eq(lowering.pipeline(), (2, 8))
  let (_, _, _, shared) = lowering.launch()
  assert_eq(shared, 66844L)
  let emitted = emit_cuda_attention_tile_source(
    lowering,
    source_test_abi(),
    "pipeline_decode",
    1048576,
  )
  let source = @utf8.decode(emitted.bytes())
  assert_true(source.contains("cp.async.ca.shared.global"))
  assert_true(source.contains("cp.async.commit_group"))
  assert_true(source.contains("cp.async.wait_group 0"))
  assert_eq(lowering.transfer_readiness(), KeyScoreBeforeValueReadiness)
  assert_true(source.contains("cp.async.wait_group 1"))
  assert_true(source.contains("scores[key / LF_SPLITS_PER_HEAD] = score"))
  assert_true(source.contains("value_cache + addresses[item]"))
  assert_true(
    source.contains("stage_tile(key_base + LF_KEY_VALUE_TILE, slot ^ 1)"),
  )
  assert_true(source.contains("if (!stage_valid[slot]) return"))
  assert_true(
    source.contains(
      "for (int key = split; key < active_keys; key += LF_SPLITS_PER_HEAD)",
    ),
  )
  assert_false(source.contains("uint2 staged_key"))
  assert_eq(
    emitted.bytes(),
    emit_cuda_attention_tile_source(
      lowering,
      source_test_abi(),
      "pipeline_decode",
      1048576,
    ).bytes(),
  )
}

///|
test "retiled pipeline does not retile logical softmax partitions" {
  let lowering = source_test_decode_lowering(candidate_id=441)
  assert_eq(lowering.tile(), (1, 32, 128))
  assert_eq(lowering.key_value_partition_grain(), 64)
  assert_eq(lowering.launch().3, 33820L)
  let source = emit_cuda_attention_tile_source(
    lowering,
    source_test_abi(),
    "small_pipeline",
    1048576,
  )
  assert_true(
    @utf8.decode(source.bytes()).contains("#define LF_KEY_VALUE_TILE 32"),
  )
  let split = source_test_decode_lowering(candidate_id=441, batch_rows=1)
  let partial = emit_cuda_grouped_split_decode_attention_tile_source(
    split,
    source_test_abi(),
    "small_partial",
    1048576,
  )
  let text = @utf8.decode(partial.bytes())
  assert_true(text.contains("const int key_begin = first_tile * 64"))
  assert_true(
    text.contains("const int tile_count = (query_position + 64) / 64"),
  )
}
