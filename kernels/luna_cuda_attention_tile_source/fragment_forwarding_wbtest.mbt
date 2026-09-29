///|
test "query-owned terminal lowering forwards single-use K V directly" {
  for id in [318, 322, 2004] {
    let lowering = source_test_lowering(true, 100000L, 2, candidate_id=Some(id))
    let fragments = lowering.prefill_fragments().unwrap()
    assert_eq(fragments.program().fold().geometry(), lowering.tile())
    let source = @utf8.decode(
      emit_cuda_attention_tile_source(
        lowering,
        source_test_abi(),
        "forward",
        1048576,
      ).bytes(),
    )
    assert_true(source.contains("lf_matrix_mma_key(score[key],a,current_k"))
    assert_true(
      source.contains("lf_matrix_mma_value(out[column],probability,current_v"),
    )
    assert_true(source.contains(".reg .b32 b<4>"))
    assert_true(
      source.contains("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {b0,b1,b2,b3}"),
    )
    assert_true(
      source.contains(
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {b0,b1,b2,b3}",
      ),
    )
    assert_true(source.contains("{b0,b2}, {%0,%1,%2,%3}"))
    assert_true(source.contains("{b1,b3}, {%4,%5,%6,%7}"))
    assert_true(source.contains("{b0,b1}, {%0,%1,%2,%3}"))
    assert_true(source.contains("{b2,b3}, {%4,%5,%6,%7}"))
    assert_false(source.contains("const auto b=lf_matrix_load(current_k"))
    assert_false(source.contains("const auto value=lf_matrix_load(current_v"))
    assert_false(source.contains("const unsigned b0 = b.x["))
    // Reusable left values and probability rounding remain explicit.
    assert_true(source.contains("const auto a=lf_matrix_load(qbuf"))
    assert_true(source.contains("probability.x[pair]=lf_probability_pair("))
    assert_true(
      source.contains("denominator[r]=denominator[r]*alpha[r]+sum[r]"),
    )
    assert_true(
      @utf8.decode(lowering.canonical_bytes()).contains(
        "fragment_representation=forward-single-use-right-v1",
      ),
    )
  }
}

///|
test "CUDA fragment binding rejects incompatible lanes vectors and matrix layout" {
  let lowering = source_test_lowering(true, 100000L, 2, candidate_id=Some(322))
  let program = lowering.prefill_fragments().unwrap().program()
  for lanes in [16, 64] {
    assert_eq(
      @luna_cuda_attention_tile_lowering.CudaPrefillFragmentBinding::new(
        program,
        subgroup_lanes=lanes,
        operand_vector_elements=8,
      ),
      None,
    )
  }
  for vector in [1, 4, 16] {
    assert_eq(
      @luna_cuda_attention_tile_lowering.CudaPrefillFragmentBinding::new(
        program,
        subgroup_lanes=32,
        operand_vector_elements=vector,
      ),
      None,
    )
  }
  for (rows, columns, quantum) in [(8, 16, 16), (16, 8, 16), (16, 16, 8)] {
    let other = program
      .fold()
      .fragment_products(
        fragment_rows=rows,
        fragment_columns=columns,
        reduction_quantum=quantum,
      )
      .unwrap()
    assert_eq(
      @luna_cuda_attention_tile_lowering.CudaPrefillFragmentBinding::new(
        other,
        subgroup_lanes=32,
        operand_vector_elements=8,
      ),
      None,
    )
  }
  assert_true(
    source_test_lowering(true, 100000L, 2, candidate_id=Some(310)).prefill_fragments()
    is None,
  )
}
