///|
test "prefill streams ordered pairs and factors PV addresses without joint accumulators" {
  for id in [318, 322, 2004, 10644, 20644] {
    let lowering = source_test_lowering(true, 100000L, 2, candidate_id=Some(id))
    let binding = lowering.prefill_fragments().unwrap()
    let packing = binding.probabilities()
    assert_eq(packing.geometry(), (4, 2))
    assert_eq(packing.peak_scalar_inputs(), 2)
    let source = @utf8.decode(
      emit_cuda_attention_tile_source(
        lowering,
        source_test_abi(),
        "probability_stream",
        1048576,
      ).bytes(),
    )
    assert_false(source.contains("float weight[8]"))
    assert_false(
      source.contains("lf_matrix_mma_value(out[column],probability,current_v"),
    )
    assert_true(
      source.contains("value_origin+((column*16)^value_column_xor)*2"),
    )
    assert_true(
      source.contains(
        "lf_matrix_mma_value(out[column],probability,value_address)",
      ),
    )
    assert_true(source.contains("sum[0]+=low"))
    assert_true(source.contains("sum[1]+=high"))
    for word in 0..<4 {
      assert_true(
        source.contains("probability.x[\{word}]=lf_probability_pair(low,high)"),
      )
    }
    assert_false(source.contains("exp2"))
    assert_false(source.contains("fmaf("))
    assert_true(
      source.contains("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16"),
    )
    assert_true(
      source.contains("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32"),
    )
  }
}

///|
test "approximate exponential source requires distinct admitted candidate" {
  let lowering = source_test_lowering(
    true,
    100000L,
    2,
    candidate_id=Some(30322),
    approximate_exponential=true,
  )
  let source = @utf8.decode(
    emit_cuda_attention_tile_source(
      lowering,
      source_test_abi(),
      "probability_stream_exp2_v1",
      1048576,
    ).bytes(),
  )
  assert_true(
    source.contains("__exp2f(__fmul_rn(delta, 1.4426950408889634074f))"),
  )
  assert_false(source.contains("expf(delta)"))
  assert_false(source.contains("--use_fast_math"))
  assert_false(source.contains("fmaf("))
  assert_true(source.contains("sum[0]+=low"))
  assert_true(source.contains("cvt.rn.bf16x2.f32"))
}

///|
test "factored matrix lane addresses equal complete row XOR layout" {
  for dimensions in [64, 128] {
    let rotations = if dimensions / 8 % 8 == 0 { 8 } else { 4 }
    let layout = @physical_tile_ir.OperandLayout::new(
      columns=dimensions,
      vector=8,
      rotations~,
      row_group=8 / rotations,
    ).unwrap()
    for key in 0..<8 {
      for lane in 0..<32 {
        let row = key * 16 + lane % 16
        let (origin, mask) = layout.factor_rows().row(row).unwrap()
        let column_xor = (lane / 16 * 8) ^ mask
        for column in 0..<(dimensions / 16) {
          assert_eq(
            origin + ((column * 16) ^ column_xor),
            layout.offset(row, column * 16 + lane / 16 * 8).unwrap(),
          )
        }
      }
    }
  }
}
