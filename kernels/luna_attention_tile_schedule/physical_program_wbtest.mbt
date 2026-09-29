///|
test "physical transport refinement uses explicit capability without changing numeric law" {
  let schedule = schedule_attention_tile_program(schedule_test_program())
  guard schedule.physical_program(
      supports_immediate_copy=false,
      fragment_rows=16,
      factor_shared_key=true,
    )
    is SharedScoreProgram(cooperative) else {
    fail("expected shared-score physical program")
  }
  guard schedule.physical_program(
      supports_immediate_copy=true,
      fragment_rows=16,
      factor_shared_key=true,
    )
    is SharedScoreProgram(immediate) else {
    fail("expected shared-score physical program")
  }
  assert_eq(cooperative.transfer_mode(), CooperativeTransfer)
  assert_eq(immediate.transfer_mode(), ImmediateTransfer)
  assert_true(cooperative.matrix_probability())
  assert_eq(cooperative.matrix_probability(), immediate.matrix_probability())
  assert_eq(cooperative.query_offsets(), immediate.query_offsets())
  assert_eq(cooperative.fold_steps(), immediate.fold_steps())
  assert_eq(cooperative.terminal(), immediate.terminal())
  for
    point in [
      InitializeScoreAccumulators,
      InitializeScoreOperands,
      ValidateScoreOperands,
    ] {
    assert_eq(cooperative.bootstrap(point), immediate.bootstrap(point))
  }
}

///|
test "physical shared-key factoring depends on fragment geometry and rewrite choice" {
  let schedule = schedule_attention_tile_program(schedule_test_program())
  for
    (rows, enabled, expected) in [
      (16, true, true),
      (64, true, false),
      (16, false, false),
    ] {
    guard schedule.physical_program(
        supports_immediate_copy=false,
        fragment_rows=rows,
        factor_shared_key=enabled,
      )
      is SharedScoreProgram(program) else {
      fail("expected shared-score physical program")
    }
    assert_eq(program.factor_shared_key(), expected)
  }
}

///|
test "physical program rejects invalid target geometry instead of guessing" {
  let schedule = schedule_attention_tile_program(schedule_test_program())
  for rows in [0, -1] {
    let rejected = try {
      ignore(
        schedule.physical_program(
          supports_immediate_copy=false,
          fragment_rows=rows,
          factor_shared_key=false,
        ),
      )
      false
    } catch {
      InvalidAttentionTileSchedule(InvalidTile) => true
      _ => false
    }
    assert_true(rejected)
  }
}
