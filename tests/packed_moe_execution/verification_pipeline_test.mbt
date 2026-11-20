///|
fn fixture_state_transaction(
  value : Ref[Int32],
  saves : Ref[Int32],
  restores : Ref[Int32],
  keeps : Ref[Int32],
) -> @stage_control.StageStateTransaction {
  let backup = Ref(0)
  let phase = Ref(0)
  @stage_control.StageStateTransaction::new(
    begin_save=() => {
      assert_eq(phase.val, 0)
      backup.val = value.val
      phase.val = 1
      saves.val += 1
    },
    poll_save=() => {
      assert_eq(phase.val, 1)
      phase.val = 2
      true
    },
    commit=() => {
      assert_eq(phase.val, 2)
      phase.val = 0
      keeps.val += 1
    },
    begin_restore=() => {
      assert_eq(phase.val, 2)
      phase.val = 3
      restores.val += 1
    },
    poll_restore=() => {
      assert_eq(phase.val, 3)
      value.val = backup.val
      phase.val = 0
      true
    },
  )
}

///|
async test "two-rank verification commits all accepted inputs before exposing target tokens" {
  run_verification_pipeline_case([5, 6, 9], 3)
}

///|
async test "resident verification session survives distinct request prefill and rejection replay" {
  run_verification_pipeline_case([5, 6, 9], 3, reuse=true)
}

///|
async test "two-rank verification restores both ranks and replays exactly a rejected prefix" {
  run_verification_pipeline_case([5, 9, 0], 2)
  run_verification_pipeline_case([9, 0, 0], 1)
}

///|
async test "two-rank verification rolls back fully matched suffixes truncated by stop or length" {
  run_verification_pipeline_case([5, 6, 9], 1, remaining=1)
  run_verification_pipeline_case([5, 6, 9], 1, stop=5)
}

///|
async test "failed target execution poisons both ranks instead of ordinary rejection" {
  run_verification_pipeline_case([5, 16, 9], 0, failure=true)
}

///|
async test "failed accepted-prefix replay never exposes target tokens" {
  run_verification_pipeline_case([5, 9, 0], 2, fail_replay=true)
}

///|
async test "resident verification cancels preparation execution restore and commit without unloading" {
  for boundary in 0..<5 {
    run_verification_pipeline_case(
      if boundary == 3 {
        [5, 6, 9]
      } else {
        [5, 9, 0]
      },
      if boundary == 3 {
        3
      } else {
        2
      },
      reuse=true,
      cancel_at=boundary,
    )
  }
}

///|
async test "request cancellation cannot recover a failed target rank" {
  run_verification_pipeline_case([5, 16, 9], 0, failure=true, cancel_at=0)
}
