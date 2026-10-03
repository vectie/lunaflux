///|
test "grouped matrix decode alternative propagates heads tails and operand rings" {
  for id in [480, 481, 482, 483] {
    let lowering = source_test_decode_lowering(
      candidate_id=id,
      alternative=true,
    )
    guard lowering.physical_program() is GroupedMatrixKeyProgram(fold, effects) else {
      fail("matrix decode program was not selected")
    }
    assert_eq(fold.head_rows(), 4)
    assert_eq(lowering.compiler_register_ceiling(92), 255)
    assert_eq(effects.independently_refills(), id >= 482)
    let source = emit_cuda_attention_tile_source(
      lowering,
      source_test_abi(),
      "matrix_decode",
      1048576,
    )
    let text = @utf8.decode(source.bytes())
    assert_true(
      text.contains("numeric_law=grouped-head-matrix-bf16-probability-v1"),
    )
    assert_true(text.contains("mma.sync.aligned.m16n8k16"))
    assert_true(text.contains("if(h<LF_HEAD_ROWS)"))
    assert_true(text.contains("p1=lane/4+8<LF_HEAD_ROWS?query_position:-1"))
    assert_true(text.contains("if(position<key_limit)") || id >= 482)
    assert_false(text.contains("__exp2f("))
    assert_eq(text.contains("cp.async.ca.shared.global"), id >= 482)
    assert_eq(text.contains("int *stage_valid=tile_valid;"), id >= 482)
    assert_eq(text.split("float next[2]").length(), 2)
    assert_eq(text.split("float alpha[2], sum[2]").length(), 2)
    assert_true(
      text.contains("lf_operand_index(key, component, LF_HEAD_DIMENSION)") ||
      id < 482,
    )
    assert_eq(lowering.launch().2, 32)
    try source_test_decode_lowering(candidate_id=id) catch {
      _ => ()
    } noraise {
      _ => fail("strict F32 policy admitted BF16 probability fold")
    }
  }
}

///|
test "grouped matrix partition state retains exact head ownership" {
  let lowering = source_test_decode_lowering(
    candidate_id=483,
    alternative=true,
    batch_rows=1,
    parallel_workgroup_target=64,
  )
  assert_true(lowering.key_value_partitions() > 1)
  let source = emit_cuda_partitioned_attention_tile_source(
    lowering,
    source_test_abi(),
    "matrix_partial",
    "matrix_merge",
    16777216L,
    1048576,
  )
  let text = @utf8.decode(source.bytes())
  assert_true(text.contains("workspace[state]=CUDART_NAN_F"))
  assert_true(text.contains("workspace[state]=maximum[item%4/2]"))
  assert_true(text.contains("if(h<LF_HEAD_ROWS)"))
}
