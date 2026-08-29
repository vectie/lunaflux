///|
fn completed_event(
  request_ordinal : Int,
  submitted_at_ns : Int64,
  admitted_at_ns : Int64,
  first_token_at_ns : Int64?,
  terminal_at_ns : Int64,
  input_tokens : Int64,
  generated_tokens : Int64,
) -> @evidence.BenchmarkRequestEvent raise {
  @evidence.BenchmarkRequestEvent::new(
    request_ordinal~,
    outcome=Completed,
    submitted_at_ns~,
    admitted_at_ns=Some(admitted_at_ns),
    first_token_at_ns~,
    terminal_at_ns~,
    input_tokens~,
    generated_tokens~,
  )
}

///|
fn summary_fixture_events() -> Array[@evidence.BenchmarkRequestEvent] raise {
  [
    completed_event(0, 100L, 110L, Some(150L), 200L, 10L, 3L),
    @evidence.BenchmarkRequestEvent::new(
      request_ordinal=1,
      outcome=Rejected,
      submitted_at_ns=101L,
      admitted_at_ns=None,
      first_token_at_ns=None,
      terminal_at_ns=105L,
      input_tokens=20L,
      generated_tokens=0L,
    ),
    @evidence.BenchmarkRequestEvent::new(
      request_ordinal=2,
      outcome=TimedOut,
      submitted_at_ns=102L,
      admitted_at_ns=Some(120L),
      first_token_at_ns=None,
      terminal_at_ns=302L,
      input_tokens=30L,
      generated_tokens=0L,
    ),
    @evidence.BenchmarkRequestEvent::new(
      request_ordinal=3,
      outcome=Cancelled,
      submitted_at_ns=103L,
      admitted_at_ns=Some(130L),
      first_token_at_ns=Some(140L),
      terminal_at_ns=203L,
      input_tokens=40L,
      generated_tokens=2L,
    ),
    @evidence.BenchmarkRequestEvent::new(
      request_ordinal=4,
      outcome=Failed,
      submitted_at_ns=104L,
      admitted_at_ns=None,
      first_token_at_ns=None,
      terminal_at_ns=204L,
      input_tokens=50L,
      generated_tokens=0L,
    ),
  ]
}

///|
fn admit_summary(
  events : ArrayView[@evidence.BenchmarkRequestEvent],
) -> @evidence.BenchmarkSummaryEvidence raise {
  @evidence.BenchmarkSummaryEvidence::admit(
    engine=LunaFlux,
    profile=Latency,
    ordinal=1,
    order_position=0,
    declared_submitted=events.length(),
    measurement=test_digest('d'),
    events~,
  )
}

///|
test "integer summaries retain every terminal outcome and known percentiles" {
  let evidence = admit_summary(summary_fixture_events())
  let summary = evidence.summary()
  assert_eq(summary.submitted(), 5)
  assert_eq(summary.completed(), 1)
  assert_eq(summary.rejected(), 1)
  assert_eq(summary.timed_out(), 1)
  assert_eq(summary.cancelled(), 1)
  assert_eq(summary.failed(), 1)
  assert_eq(summary.input_tokens(), 150L)
  assert_eq(summary.generated_tokens(), 5L)
  assert_eq(summary.elapsed_ns(), 202L)
  let queue = summary.queue_ns()
  assert_eq(queue.count(), 3)
  assert_eq(queue.minimum(), 10L)
  assert_eq(queue.p50(), 18L)
  assert_eq(queue.p95(), 27L)
  assert_eq(queue.p99(), 27L)
  assert_eq(queue.maximum(), 27L)
  assert_eq(queue.total(), 55L)
  let end_to_end = summary.end_to_end_ns()
  assert_eq(end_to_end.count(), 5)
  assert_eq(end_to_end.minimum(), 4L)
  assert_eq(end_to_end.p50(), 100L)
  assert_eq(end_to_end.p95(), 200L)
  assert_eq(end_to_end.p99(), 200L)
  assert_eq(end_to_end.maximum(), 200L)
  assert_eq(end_to_end.total(), 504L)
  let ttft = summary.ttft_ns()
  assert_eq(ttft.count(), 2)
  assert_eq(ttft.p50(), 37L)
  assert_eq(ttft.p95(), 50L)
  assert_eq(ttft.total(), 87L)
  let service = summary.service_ns()
  assert_eq(service.count(), 3)
  assert_eq(service.p50(), 90L)
  assert_eq(service.p95(), 182L)
  assert_eq(service.total(), 345L)
  let decode = summary.decode_ns()
  assert_eq(decode.count(), 2)
  assert_eq(decode.p50(), 50L)
  assert_eq(decode.p95(), 63L)
  assert_eq(decode.total(), 113L)
}

///|
test "canonical raw and summary evidence are stable and trial-consumable" {
  let first = admit_summary(summary_fixture_events())
  let second = admit_summary(summary_fixture_events())
  assert_eq(first.raw_events_digest(), second.raw_events_digest())
  assert_eq(first.summary_digest(), second.summary_digest())
  assert_eq(
    first.raw_events_canonical_bytes(),
    second.raw_events_canonical_bytes(),
  )
  assert_eq(first.summary_canonical_bytes(), second.summary_canonical_bytes())
  assert_not_eq(first.raw_events_digest(), first.summary_digest())
  let trial = @evidence.BenchmarkTrial::from_summary(
    measurements=first,
    correctness_passed=true,
    correctness=test_digest('e'),
  )
  assert_eq(trial.engine(), LunaFlux)
  assert_eq(trial.profile(), Latency)
  assert_eq(trial.ordinal(), 1)
  assert_eq(trial.order_position(), 0)
}

///|
test "measurement identity and event values are digest-bound" {
  let events = summary_fixture_events()
  let first = admit_summary(events)
  let different_measurement = @evidence.BenchmarkSummaryEvidence::admit(
    engine=LunaFlux,
    profile=Latency,
    ordinal=1,
    order_position=0,
    declared_submitted=events.length(),
    measurement=test_digest('f'),
    events~,
  )
  let changed_events = summary_fixture_events()
  changed_events[4] = @evidence.BenchmarkRequestEvent::new(
    request_ordinal=4,
    outcome=Failed,
    submitted_at_ns=104L,
    admitted_at_ns=None,
    first_token_at_ns=None,
    terminal_at_ns=205L,
    input_tokens=50L,
    generated_tokens=0L,
  )
  let changed = admit_summary(changed_events)
  assert_not_eq(
    first.raw_events_digest(),
    different_measurement.raw_events_digest(),
  )
  assert_not_eq(first.raw_events_digest(), changed.raw_events_digest())
  assert_not_eq(first.summary_digest(), changed.summary_digest())
}

///|
test "request event exposes only canonical ordinal and terminal outcome" {
  let event = completed_event(7, 10L, 11L, None, 12L, 3L, 0L)
  assert_eq(event.request_ordinal(), 7)
  assert_eq(event.outcome(), Completed)
  assert_eq(@evidence.maximum_request_events_per_summary(), 8192)
}

///|
test "unobserved timing distributions have an explicit zero representation" {
  let events = [
    @evidence.BenchmarkRequestEvent::new(
      request_ordinal=0,
      outcome=Rejected,
      submitted_at_ns=10L,
      admitted_at_ns=None,
      first_token_at_ns=None,
      terminal_at_ns=12L,
      input_tokens=1L,
      generated_tokens=0L,
    ),
  ]
  let summary = admit_summary(events).summary()
  for
    distribution in [
      summary.queue_ns(),
      summary.ttft_ns(),
      summary.service_ns(),
      summary.decode_ns(),
    ] {
    assert_eq(distribution.count(), 0)
    assert_eq(distribution.minimum(), 0L)
    assert_eq(distribution.p50(), 0L)
    assert_eq(distribution.p95(), 0L)
    assert_eq(distribution.p99(), 0L)
    assert_eq(distribution.maximum(), 0L)
    assert_eq(distribution.total(), 0L)
  }
}
