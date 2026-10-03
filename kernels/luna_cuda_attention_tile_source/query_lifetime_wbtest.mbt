///|
test "retained queries are executable distinct same-law alternatives" {
  let baseline = source_test_lowering(true, 100000L, 2, candidate_id=Some(322))
  for (id, retained) in [(10644, 64), (10645, 128)] {
    let lowering = source_test_lowering(true, 100000L, 2, candidate_id=Some(id))
    let binding = lowering.prefill_fragments().unwrap()
    let lifetime = binding.query_lifetime()
    assert_eq(lifetime.retained_dimensions(), retained)
    assert_eq(
      binding.program().fold(),
      baseline.prefill_fragments().unwrap().program().fold(),
    )
    assert_eq(lowering.launch(), baseline.launch())
    assert_false(lowering.digest() == baseline.digest())
    let first = emit_cuda_attention_tile_source(
      lowering,
      source_test_abi(),
      "retained_query",
      1048576,
    )
    assert_eq(
      first,
      emit_cuda_attention_tile_source(
        lowering,
        source_test_abi(),
        "retained_query",
        1048576,
      ),
    )
    let source = @utf8.decode(first.bytes())
    assert_true(source.contains("query_operand_0=lf_matrix_load(qbuf,qr,0"))
    assert_true(source.contains("if (qr<active && key_begin<key_end)"))
    assert_true(source.contains("lf_matrix_mma_key(score[key],query_operand_0"))
    assert_true(source.contains("lf_masked_exp(score[key].x[0]"))
    assert_true(
      source.contains("denominator[r]=denominator[r]*alpha[r]+sum[r]"),
    )
    assert_false(source.contains("exp2"))
    if retained == 128 {
      assert_false(source.contains("const auto a=lf_matrix_load(qbuf"))
    } else {
      assert_true(
        source.contains("for (int reduction=64;reduction<LF_HEAD_DIMENSION"),
      )
    }
    // Bootstrap is outside the ascending KV loop, not an inert lifetime note.
    assert_true(
      source.find("query_operand_0=lf_matrix_load").unwrap() <
      source.find("for (int kb=key_begin").unwrap(),
    )
    assert_true(
      @utf8.decode(lowering.canonical_bytes()).contains(
        "query_operand_lifetime=retained-prefix-v1,\{retained}",
      ),
    )
  }
}

///|
test "query lifetimes compose with paged dense split and transfer schedules" {
  for base in [318, 319, 320, 321, 322, 323, 324, 1003, 1004, 1005, 1006, 1008] {
    for retention in [0, 1] {
      let candidate = 10000 + base * 2 + retention
      for dense in [false, true] {
        for partitioned in [false, true] {
          let lowering = source_test_lowering(
            true,
            100000L,
            2,
            candidate_id=Some(candidate),
            dense_current=dense,
            partitioned~,
          )
          let binding = lowering.prefill_fragments().unwrap()
          assert_eq(
            binding.query_lifetime().retained_dimensions(),
            if retention == 0 {
              64
            } else {
              128
            },
          )
          let source = @utf8.decode(
            emit_cuda_query_owned_attention_source(
              lowering,
              source_test_abi(),
              "query_lifetime_effects",
              1048576,
            ).bytes(),
          )
          assert_true(source.contains("if (qr<active && key_begin<key_end)"))
          assert_true(
            source.contains("query_operand_0=lf_matrix_load(qbuf,qr,0"),
          )
          assert_true(
            source.find("query_operand_0=lf_matrix_load").unwrap() <
            source.find("for (int kb=key_begin").unwrap(),
          )
          // Retention cannot replace visibility or output-tail guards.
          assert_true(source.contains("positions[token_base+query0]"))
          assert_true(source.contains("if (query<active)"))
          assert_true(source.contains("lf_masked_exp(score[key].x[0]"))
          assert_false(source.contains("exp2"))
        }
      }
    }
  }
}
