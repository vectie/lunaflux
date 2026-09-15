///|
fn check_bootstrap_validation(
  actions : ReadOnlyArray[
    @luna_attention_tile_schedule.AttentionScoreBootstrapEffect,
  ],
  valid : Bool,
) -> Unit raise {
  let mut written = false
  let mut visible = false
  let mut reading = false
  let mut accepted = false
  for action in actions {
    match action {
      ValidateScoreQueryBounds => written = true
      PublishBootstrapWrites => {
        assert_true(written)
        visible = true
      }
      ReadScoreQueryBounds => {
        assert_true(visible)
        reading = true
      }
      RequireValidScoreQueries => {
        assert_true(visible && reading)
        if !valid {
          return
        }
        accepted = true
      }
      ReleaseScoreValidationReaders => {
        assert_true(accepted)
        reading = false
      }
      _ => fail("unexpected validation operation")
    }
  }
  assert_true(accepted && !reading)
}

///|
test "score bootstrap publishes operands and releases borrowed validation storage" {
  for resident_output in [false, true] {
    let initial = @luna_attention_tile_schedule.plan_attention_score_bootstrap(
      InitializeScoreAccumulators,
      resident_output~,
    )
    assert_eq(
      initial.iter().to_array(),
      if resident_output {
        [
          InitializeScoreOutput,
          PublishBootstrapWrites,
          InitializeResidentScoreOutput,
          PublishBootstrapWrites,
        ]
      } else {
        [InitializeScoreOutput, PublishBootstrapWrites]
      },
    )
    assert_eq(
      @luna_attention_tile_schedule.plan_attention_score_bootstrap(
        InitializeScoreOperands,
        resident_output~,
      )
      .iter()
      .to_array(),
      [StageScoreQueries, InitializeScoreStatistics, PublishBootstrapWrites],
    )
    let validation = @luna_attention_tile_schedule.plan_attention_score_bootstrap(
      ValidateScoreOperands,
      resident_output~,
    )
    check_bootstrap_validation(validation, true)
    check_bootstrap_validation(validation, false)
    for index in [1, 2, 3, 4] {
      let remaining = []
      for i, action in validation {
        if i != index {
          remaining.push(action)
        }
      }
      try
        check_bootstrap_validation(ReadOnlyArray::from_array(remaining), true)
      catch {
        _ => ()
      } noraise {
        _ => fail("missing validation effect accepted")
      }
    }
  }
}
