///|
test "joint query fanout binds one operand and preserves ordered matrix products" {
  for (id, retained) in [(20644, 64), (20645, 128)] {
    let lowering = source_test_lowering(true, 100000L, 2, candidate_id=Some(id))
    let prior = source_test_lowering(
      true,
      100000L,
      2,
      candidate_id=Some(id - 10000),
    )
    let binding = lowering.prefill_fragments().unwrap()
    let lifetime = binding.query_lifetime()
    assert_true(lifetime.joint_right_consumers())
    assert_eq(lifetime.retained_dimensions(), retained)
    assert_eq(lifetime.program(), prior.prefill_fragments().unwrap().program())
    assert_eq(
      lifetime.peak_fragment_words(),
      prior.prefill_fragments().unwrap().query_lifetime().peak_fragment_words(),
    )
    assert_eq(lowering.launch(), prior.launch())
    assert_false(lowering.digest() == prior.digest())
    let region = cuda_joint_query_consumer(binding, 0)
    assert_eq(
      region
      .split("ldmatrix.sync.aligned.m8n8.x4.shared.b16")
      .to_array()
      .length() -
      1,
      4,
    )
    assert_eq(
      region
      .split("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32")
      .to_array()
      .length() -
      1,
      8,
    )
    assert_eq(
      region.split("\"r\"(query_operand_0.x[0])").to_array().length() - 1,
      1,
    )
    assert_eq(
      region.split("add.u32 key_address, key_address, 4096").to_array().length() -
      1,
      3,
    )
    assert_true(region.contains("{b0,b2}, {%0,%1,%2,%3}"))
    assert_true(region.contains("{b1,b3}, {%4,%5,%6,%7}"))
    assert_true(region.contains("{%24,%25,%26,%27}, {%32,%33,%34,%35}"))
    let source = @utf8.decode(
      emit_cuda_attention_tile_source(
        lowering,
        source_test_abi(),
        "joint_query",
        1048576,
      ).bytes(),
    )
    assert_true(source.contains("// joint-right-columns-v1:"))
    assert_false(source.contains("lf_matrix_mma_key(score[key],query_operand_"))
    assert_true(
      source.contains("denominator[r]=denominator[r]*alpha[r]+sum[r]"),
    )
    assert_true(source.contains("lf_masked_exp(score[key].x[0]"))
    assert_false(source.contains("exp2"))
    assert_true(
      @utf8.decode(lowering.canonical_bytes()).contains(
        "query_operand_consumer=joint-right-columns-v1",
      ),
    )
    let old_source = @utf8.decode(
      emit_cuda_attention_tile_source(
        prior,
        source_test_abi(),
        "joint_query",
        1048576,
      ).bytes(),
    )
    assert_false(old_source.contains("// joint-right-columns-v1:"))
  }
}

///|
test "joint shared row increment preserves XOR for every matrix lane" {
  // This is the emitted CUDA layout formula, checked over all legal matrix
  // dimensions and key row boundaries rather than a Qwen-only sample.
  for dimensions in [64, 128] {
    for column in 0..<dimensions {
      let vectors = dimensions / 8
      let rotations = if vectors % 8 == 0 {
        8
      } else if vectors % 4 == 0 {
        4
      } else {
        2
      }
      for row in 0..<112 {
        let first = row * dimensions +
          (column ^ (row / (8 / rotations) % rotations * 8))
        let next = (row + 16) * dimensions +
          (column ^ ((row + 16) / (8 / rotations) % rotations * 8))
        assert_eq(next - first, 16 * dimensions)
      }
    }
  }
}
