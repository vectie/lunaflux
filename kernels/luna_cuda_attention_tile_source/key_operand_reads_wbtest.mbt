///|
test "contiguous QK reads propagate through strategy schedule lowering and source" {
  for (baseline_id, strict) in [(322, true), (30322, false)] {
    let base = source_test_lowering(
      true,
      100000L,
      2,
      candidate_id=Some(baseline_id),
      approximate_exponential=!strict,
    )
    for slots in [1, 2] {
      let id = (slots + 1) * 100000 + baseline_id
      let lowering = source_test_lowering(
        true,
        100000L,
        2,
        candidate_id=Some(id),
        approximate_exponential=!strict,
      )
      let binding = lowering.prefill_fragments().unwrap()
      assert_eq(lowering.launch(), base.launch())
      assert_eq(binding.program(), base.prefill_fragments().unwrap().program())
      assert_eq(binding.key_operand_reads().unwrap().geometry(), (4, slots))
      assert_eq(binding.key_read_ownership().unwrap().provider(8), Some(16))
      assert_eq(binding.score().word(0, 1), Some(1))
      let source = @utf8.decode(
        emit_cuda_attention_tile_source(
          lowering,
          source_test_abi(),
          "reads",
          1048576,
        ).bytes(),
      )
      assert_true(
        @utf8.decode(lowering.canonical_bytes()).contains(
          "right_operand_reads=consumer-contiguous-v1,\{slots}",
        ),
      )
      assert_true(
        source.contains("denominator[r]=denominator[r]*alpha[r]+sum[r]"),
      )
      if slots == 1 {
        assert_true(source.contains("row + (lane & 7) + lane / 16 * 8"))
        assert_true(source.contains("{b0,b1}, {%0,%1,%2,%3}"))
        assert_false(source.contains("lf_key_operand_pipeline"))
      } else {
        assert_true(
          source.contains(
            "lf_key_operand_pipeline(score,a,current_k,reduction);",
          ),
        )
        assert_true(source.contains(".reg .b32 b<8>"))
        assert_true(source.contains("{b6,b7}"))
        assert_false(source.contains("lf_matrix_mma_key"))
        let helper = cuda_key_operand_read_pipeline(binding)
        assert_eq(
          helper.split("ldmatrix.sync.aligned.m8n8.x4").to_array().length() - 1,
          4,
        )
        assert_eq(
          helper.split("mma.sync.aligned.m16n8k16").to_array().length() - 1,
          8,
        )
        assert_eq(
          helper.split("add.u32 address, address").to_array().length() - 1,
          3,
        )
      }
      assert_false(lowering.digest() == base.digest())
    }
  }
}

///|
test "read-ahead covers complete legal columns and rejects incompatible lifetime" {
  for id in [300319, 300322, 302004] {
    let binding = source_test_lowering(true, 100000L, 2, candidate_id=Some(id))
      .prefill_fragments()
      .unwrap()
    let (columns, slots) = binding.key_operand_reads().unwrap().geometry()
    assert_eq(slots, 2)
    let helper = cuda_key_operand_read_pipeline(binding)
    assert_eq(
      helper.split("ldmatrix.sync.aligned.m8n8.x4").to_array().length() - 1,
      columns,
    )
  }
  let program = source_test_lowering(true, 100000L, 2, candidate_id=Some(322))
    .prefill_fragments()
    .unwrap()
    .program()
  for
    (slots, retained, joint) in [
      (-1, 0, false),
      (3, 0, false),
      (1, 64, false),
      (2, 0, true),
    ] {
    assert_eq(
      @luna_cuda_attention_tile_lowering.CudaPrefillFragmentBinding::new(
        program,
        subgroup_lanes=32,
        operand_vector_elements=8,
        right_operand_read_slots=slots,
        retained_query_dimensions=retained,
        joint_query_consumers=joint,
      ),
      None,
    )
  }
}
