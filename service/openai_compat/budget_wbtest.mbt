///|
fn openai_wb_collect(
  workspace : LunaOpenAICompatWorkspace,
  event : @luna_event.LunaEventView,
  mode : LunaOpenAICompatMode,
  sequence : UInt64,
) -> (FixedArray[Byte], UInt64) raise {
  let work = workspace.begin(event, mode, response_sequence=sequence)
  while work.state() == LunaOpenAICompatPending {
    assert_eq(work.progress(), LunaOpenAICompatAdvanced)
    assert_true(work.last_work_units() > 0)
    assert_true(work.last_work_units() <= workspace.budget.work_units())
  }
  let total = work.total_work_units()
  let view = work.take_view()
  let result : FixedArray[Byte] = FixedArray::make(view.length(), 0)
  let mut offset = 0
  while offset < result.length() {
    let remaining = result.length() - offset
    let length = if remaining < workspace.budget.work_units() {
      remaining
    } else {
      workspace.budget.work_units()
    }
    ignore(
      view.copy_chunk_to(
        result,
        destination_offset=offset,
        source_offset=offset,
        length~,
      ),
    )
    offset += length
  }
  view.release()
  (result, total)
}

///|
fn openai_wb_assert_same(
  left : FixedArray[Byte],
  right : FixedArray[Byte],
) -> Unit raise {
  assert_eq(left.length(), right.length())
  for index in 0..<left.length() {
    assert_eq(left[index], right[index])
  }
}

///|
test "Luna OpenAI budgets 1 17 and maximum preserve bytes and exact work" {
  let limits = openai_wb_fixture_limits()
  let modes : FixedArray[LunaOpenAICompatMode] = [
    LunaOpenAIResponses,
    LunaOpenAIChatCompletions,
  ]
  for mode in modes {
    let mut baseline : FixedArray[Byte] = []
    let mut baseline_total = 0UL
    for budget in [1, 17, 65536] {
      let (owner, event) = openai_token_event(limits)
      let (actual, total) = openai_wb_collect(
        openai_wb_workspace(limits, budget),
        event,
        mode,
        18446744073709551615UL,
      )
      if baseline.length() == 0 {
        baseline = actual
        baseline_total = total
      } else {
        openai_wb_assert_same(actual, baseline)
        assert_eq(total, baseline_total)
      }
      owner.retire(event)
    }
  }
}

///|
test "Luna OpenAI every progress call debits no more than its budget" {
  let limits = openai_wb_fixture_limits()
  let (owner, event) = openai_token_event(limits)
  let workspace = openai_wb_workspace(limits, 17)
  let work = workspace.begin(event, LunaOpenAIResponses, response_sequence=7UL)
  let mut summed = 0UL
  while work.state() == LunaOpenAICompatPending {
    assert_eq(work.progress(), LunaOpenAICompatAdvanced)
    assert_true(work.last_work_units() > 0)
    assert_true(work.last_work_units() <= 17)
    summed += work.last_work_units().to_uint64()
    assert_eq(work.total_work_units(), summed)
  }
  assert_eq(work.progress(), LunaOpenAICompatIdle)
  assert_eq(work.last_work_units(), 0)
  assert_eq(work.total_work_units(), summed)
  work.abort()
  owner.retire(event)
}
