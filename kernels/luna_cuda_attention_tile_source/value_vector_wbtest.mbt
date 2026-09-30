///|
test "packed value ownership reaches ordinary and partitioned epilogues" {
  let ordinary = source_test_decode_lowering(candidate_id=452, alternative=true)
  let partitioned = source_test_decode_lowering(
    candidate_id=452,
    alternative=true,
    batch_rows=1,
    parallel_workgroup_target=96,
  )
  for lowering in [ordinary, partitioned] {
    let text = if lowering.key_value_partitions() == 1 {
      @utf8.decode(
        emit_cuda_attention_tile_source(
          lowering,
          source_test_abi(),
          "decode",
          1048576,
        ).bytes(),
      )
    } else {
      @utf8.decode(
        emit_cuda_partitioned_attention_tile_source(
          lowering,
          source_test_abi(),
          "partial",
          "merge",
          16777216L,
          1048576,
        ).bytes(),
      )
    }
    assert_true(
      text.contains("#define LF_VALUE_COMPONENT(item) (lane * 4 + item)"),
    )
    assert_true(text.contains("const uint2 packed ="))
    assert_true(
      text.contains("const int component = LF_VALUE_COMPONENT(item);"),
    )
    assert_true(text.contains("numeric_law=blockwise-f32-probability-v1"))
    assert_true(text.contains("[%0], [%1], 16, %2;"))
    assert_false(text.contains("fmaf("))
  }
}
