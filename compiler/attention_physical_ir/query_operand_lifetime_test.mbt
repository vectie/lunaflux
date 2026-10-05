///|
test "query retention preserves fold laws across complete and ragged epochs" {
  for queries in [16, 32, 64] {
    for keys in [32, 64, 128] {
      for dimensions in [64, 128] {
        let fold = @attention_physical_ir.OnlineFold::new(
          queries~,
          keys~,
          dimensions~,
        ).unwrap()
        let fragments = fold
          .fragment_products(
            fragment_rows=16,
            fragment_columns=16,
            reduction_quantum=16,
          )
          .unwrap()
        let original = @attention_physical_ir.QueryOperandLifetime::new(
          fragments,
          retained_dimensions=0,
          operand_words=4,
          accumulator_words=8,
        ).unwrap()
        for retained in [0, dimensions / 2, dimensions] {
          let plan = @attention_physical_ir.QueryOperandLifetime::new(
            fragments,
            retained_dimensions=retained,
            operand_words=4,
            accumulator_words=8,
          ).unwrap()
          assert_eq(plan.program().fold(), fold)
          assert_eq(plan.program().score(), fragments.score())
          assert_eq(plan.program().value(), fragments.value())
          for history in [0, 1, keys - 1, keys, keys + 1, 4095] {
            let epochs = if history == 0 { 0 } else { 1 + (history - 1) / keys }
            let loads = plan.query_load_count(epochs).unwrap()
            assert_true(loads <= original.query_load_count(epochs).unwrap())
            if retained == dimensions && epochs > 0 {
              assert_eq(loads, (dimensions / 16).to_int64())
            }
          }
          assert_true(
            plan.peak_fragment_words() >= original.peak_fragment_words(),
          )
        }
      }
    }
  }
}

///|
test "joint operand fanout refines consumption without enlarging fragment state" {
  let program = @attention_physical_ir.OnlineFold::new(
      queries=64,
      keys=64,
      dimensions=128,
    )
    .unwrap()
    .fragment_products(
      fragment_rows=16,
      fragment_columns=16,
      reduction_quantum=16,
    )
    .unwrap()
  for retained in [0, 64, 128] {
    let prior = @attention_physical_ir.QueryOperandLifetime::new(
      program,
      retained_dimensions=retained,
      operand_words=4,
      accumulator_words=8,
    ).unwrap()
    let joint = @attention_physical_ir.QueryOperandLifetime::new(
      program,
      retained_dimensions=retained,
      operand_words=4,
      accumulator_words=8,
      joint_right_consumers=true,
    ).unwrap()
    assert_true(joint.joint_right_consumers())
    assert_eq(prior.program(), joint.program())
    assert_eq(prior.peak_fragment_words(), joint.peak_fragment_words())
    assert_eq(prior.query_load_count(128), joint.query_load_count(128))
  }
}

///|
test "query retention rejects incomplete and mismatched lifetime extents" {
  let program = @attention_physical_ir.OnlineFold::new(
      queries=64,
      keys=64,
      dimensions=128,
    )
    .unwrap()
    .fragment_products(
      fragment_rows=16,
      fragment_columns=16,
      reduction_quantum=16,
    )
    .unwrap()
  for retained in [-1, 1, 15, 17, 129] {
    assert_eq(
      @attention_physical_ir.QueryOperandLifetime::new(
        program,
        retained_dimensions=retained,
        operand_words=4,
        accumulator_words=8,
      ),
      None,
    )
  }
  for words in [0, -1, 1025] {
    assert_eq(
      @attention_physical_ir.QueryOperandLifetime::new(
        program,
        retained_dimensions=64,
        operand_words=words,
        accumulator_words=8,
      ),
      None,
    )
  }
  assert_eq(
    @attention_physical_ir.QueryOperandLifetime::new(
      program,
      retained_dimensions=64,
      operand_words=4,
      accumulator_words=8,
    )
    .unwrap()
    .query_load_count(-1),
    None,
  )
}
