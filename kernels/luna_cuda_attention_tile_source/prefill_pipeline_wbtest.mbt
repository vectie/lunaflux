///|
test "matrix prefill overlaps lookahead copies with an unchanged ordered fold" {
  for candidate_id in [316, 317] {
    let lowering = source_test_lowering(
      true,
      100000L,
      2,
      candidate_id=Some(candidate_id),
    )
    assert_eq(lowering.copy(), CudaAsynchronousSharedLoad)
    assert_eq(lowering.pipeline(), (2, 16))
    let emitted = emit_cuda_attention_tile_source(
      lowering,
      source_test_abi(),
      "prefill_async",
      1048576,
    )
    assert_eq(
      emitted,
      emit_cuda_attention_tile_source(
        lowering,
        source_test_abi(),
        "prefill_async",
        1048576,
      ),
    )
    let source = @utf8.decode(emitted.bytes())
    for
      required in [
        "#define LF_ASYNC_PREFILL 1", "cp.async.cg.shared.global", "cp.async.commit_group;",
        "cp.async.wait_group 0;", "stage_prefill_tile(partition_begin, 0)", "stage_prefill_tile(next_key_base, stage ^ 1)",
        "const int source_bytes = page >= 0 ? 16 : 0", "fold_denominator_0 * previous_scale + tile_sum",
        "lf_transformed_score",
      ] {
      assert_true(source.contains(required))
    }
    let layout = lowering.matrix_storage_layout().unwrap()
    assert_true(layout.offset(ProbabilityStorage) != layout.offset(KeyStorage))
    assert_true(
      layout.offset(ValidationStorage) > layout.offset(AccumulatorStorage),
    )
  }
}

///|
test "synchronous matrix prefill retains its compact storage and no async issuer" {
  let lowering = source_test_lowering(false, 100000L, 2, candidate_id=Some(312))
  let source = @utf8.decode(
    emit_cuda_attention_tile_source(
      lowering,
      source_test_abi(),
      "prefill_sync",
      1048576,
    ).bytes(),
  )
  assert_true(source.contains("#define LF_ASYNC_PREFILL 0"))
  assert_false(source.contains("cp.async.cg.shared.global"))
  let layout = lowering.matrix_storage_layout().unwrap()
  assert_eq(layout.offset(ProbabilityStorage), layout.offset(KeyStorage))
}
