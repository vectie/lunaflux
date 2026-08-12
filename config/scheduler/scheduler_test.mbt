///|
fn scheduler(
  step_token_budget? : Int = 128,
  max_active_requests? : Int = 32,
  max_waiting_requests? : Int = 64,
  prefill_chunk_tokens? : Int = 64,
  emergency_decode_page_reserve? : Int = 8,
  waiting_age_threshold_steps? : Int = 128,
) -> @scheduler.SchedulerConfig raise @scheduler.SchedulerConfigError {
  @scheduler.new(
    step_token_budget~,
    max_active_requests~,
    max_waiting_requests~,
    prefill_chunk_tokens~,
    emergency_decode_page_reserve~,
    waiting_age_threshold_steps~,
  )
}

///|
fn expect_scheduler_error(
  run : () -> @scheduler.SchedulerConfig raise @scheduler.SchedulerConfigError,
  expected : @scheduler.SchedulerConfigError,
) -> Unit raise {
  try run() catch {
    error => assert_eq(error, expected)
  } noraise {
    _ => fail("expected scheduler configuration rejection")
  }
}

///|
test "scheduler preserves exact values and inclusive minima" {
  let config = scheduler(
    step_token_budget=1,
    max_active_requests=1,
    max_waiting_requests=1,
    prefill_chunk_tokens=1,
    emergency_decode_page_reserve=1,
    waiting_age_threshold_steps=1,
  )
  assert_eq(config.step_token_budget(), 1)
  assert_eq(config.max_active_requests(), 1)
  assert_eq(config.max_waiting_requests(), 1)
  assert_eq(config.prefill_chunk_tokens(), 1)
  assert_eq(config.emergency_decode_page_reserve(), 1)
  assert_eq(config.waiting_age_threshold_steps(), 1)
  assert_eq(config.plan_buffer_count(), 2)
}

///|
test "scheduler accepts every inclusive hard ceiling" {
  let config = scheduler(
    step_token_budget=1_048_576,
    max_active_requests=65_536,
    max_waiting_requests=983_040,
    prefill_chunk_tokens=262_144,
    emergency_decode_page_reserve=65_536,
    waiting_age_threshold_steps=1_000_000,
  )
  assert_eq(config.step_token_budget(), 1_048_576)
  assert_eq(config.max_active_requests(), 65_536)
  assert_eq(config.max_waiting_requests(), 983_040)
  assert_eq(config.prefill_chunk_tokens(), 262_144)
  assert_eq(config.emergency_decode_page_reserve(), 65_536)
  assert_eq(config.waiting_age_threshold_steps(), 1_000_000)
  let waiting_ceiling = scheduler(
    step_token_budget=48_576,
    max_active_requests=48_576,
    max_waiting_requests=1_000_000,
    emergency_decode_page_reserve=48_576,
  )
  assert_eq(waiting_ceiling.max_waiting_requests(), 1_000_000)
}

///|
test "scheduler keeps concurrency and page reserve independent of one step" {
  let config = scheduler(
    step_token_budget=8,
    max_active_requests=32,
    emergency_decode_page_reserve=64,
    prefill_chunk_tokens=8,
  )
  assert_eq(config.step_token_budget(), 8)
  assert_eq(config.max_active_requests(), 32)
  assert_eq(config.emergency_decode_page_reserve(), 64)
}

///|
test "scheduler rejects every numeric lower bound" {
  expect_scheduler_error(
    () => scheduler(step_token_budget=0),
    Invalid(field=StepTokenBudget, issue=BelowMinimum),
  )
  expect_scheduler_error(
    () => scheduler(max_active_requests=0),
    Invalid(field=MaxActiveRequests, issue=BelowMinimum),
  )
  expect_scheduler_error(
    () => scheduler(max_waiting_requests=0),
    Invalid(field=MaxWaitingRequests, issue=BelowMinimum),
  )
  expect_scheduler_error(
    () => scheduler(prefill_chunk_tokens=0),
    Invalid(field=PrefillChunkTokens, issue=BelowMinimum),
  )
  expect_scheduler_error(
    () => scheduler(emergency_decode_page_reserve=0),
    Invalid(field=EmergencyDecodePageReserve, issue=BelowMinimum),
  )
  expect_scheduler_error(
    () => scheduler(waiting_age_threshold_steps=0),
    Invalid(field=WaitingAgeThresholdSteps, issue=BelowMinimum),
  )
}

///|
test "scheduler rejects every numeric hard-ceiling breach" {
  expect_scheduler_error(
    () => scheduler(step_token_budget=1_048_577),
    Invalid(field=StepTokenBudget, issue=AboveMaximum),
  )
  expect_scheduler_error(
    () => scheduler(step_token_budget=65_537, max_active_requests=65_537),
    Invalid(field=MaxActiveRequests, issue=AboveMaximum),
  )
  expect_scheduler_error(
    () => scheduler(max_waiting_requests=1_000_001),
    Invalid(field=MaxWaitingRequests, issue=AboveMaximum),
  )
  expect_scheduler_error(
    () => scheduler(step_token_budget=262_145, prefill_chunk_tokens=262_145),
    Invalid(field=PrefillChunkTokens, issue=AboveMaximum),
  )
  expect_scheduler_error(
    () => {
      scheduler(
        step_token_budget=65_537,
        max_active_requests=65_536,
        emergency_decode_page_reserve=65_537,
      )
    },
    Invalid(field=EmergencyDecodePageReserve, issue=AboveMaximum),
  )
  expect_scheduler_error(
    () => scheduler(waiting_age_threshold_steps=1_000_001),
    Invalid(field=WaitingAgeThresholdSteps, issue=AboveMaximum),
  )
}

///|
test "scheduler rejects every internal relation breach" {
  expect_scheduler_error(
    () => scheduler(step_token_budget=63, prefill_chunk_tokens=64),
    Invalid(field=PrefillChunkTokens, issue=ExceedsStepTokenBudget),
  )
  expect_scheduler_error(
    () => {
      scheduler(
        step_token_budget=65_536,
        max_active_requests=65_536,
        max_waiting_requests=983_041,
      )
    },
    Invalid(field=MaxWaitingRequests, issue=CombinedRequestCapacityExceeded),
  )
}
