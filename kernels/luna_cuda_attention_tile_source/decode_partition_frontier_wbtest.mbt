///|
test "all independent and matrix decode candidates reach partitioned source unconditionally" {
  let abi = CudaPagedAttentionTileAbi::new(
    query_heads=16,
    key_value_heads=8,
    head_dimension=128,
    tokens_per_page=8,
    input_row_width=4096,
    page_stride_values=8192L,
    physical_page_capacity=16384L,
    maximum_query_rows=32,
    maximum_query_tokens=2048,
    maximum_page_entries=16384,
  )
  for
    id in [
      460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 480, 481, 482,
      483,
    ] {
    for target in [16, 32, 64] {
      let lowering = source_test_decode_lowering(
        candidate_id=id,
        alternative=true,
        batch_rows=1,
        parallel_workgroup_target=target,
        key_value_heads=8,
        tokens_per_page=8,
      )
      assert_eq(lowering.key_value_partitions(), target / 8)
      assert_eq(lowering.launch().1, 8)
      let source = emit_cuda_partitioned_attention_tile_source(
        lowering, abi, "frontier_partial", "frontier_merge", 16777216L, 1048576,
      )
      assert_eq(source.symbols(), ("frontier_partial", "frontier_merge"))
      assert_eq(
        source.workspace().partial_state_count(),
        32L * 16L * (target / 8).to_int64(),
      )
      let text = @utf8.decode(source.bytes())
      assert_true(
        text.contains("extern \"C\" __global__ void frontier_partial("),
      )
      assert_true(text.contains("extern \"C\" __global__ void frontier_merge("))
      if id < 480 || id >= 482 {
        assert_true(
          text.contains("stage_key(key_begin, 0); stage_value(key_begin, 0);"),
        )
        assert_true(text.contains("cp.async.wait_group 3;"))
      }
    }
  }
}
