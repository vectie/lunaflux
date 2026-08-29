///|
fn expect_invalid_timestamp(
  build : () -> @evidence.BenchmarkRequestEvent raise,
) -> Unit raise {
  try build() catch {
    @evidence.InvalidEventTimestamp => ()
    _ => fail("unexpected timestamp error")
  } noraise {
    _ => fail("non-monotonic request timestamps were admitted")
  }
}

///|
test "request-local timestamps must be monotonic" {
  expect_invalid_timestamp(() => completed_event(0, 10L, 11L, None, 9L, 1L, 0L))
  expect_invalid_timestamp(() => completed_event(0, 10L, 13L, None, 12L, 1L, 0L))
  expect_invalid_timestamp(() => {
    completed_event(0, 10L, 12L, Some(11L), 13L, 1L, 1L)
  })
}

///|
test "generated output and terminal outcome shape fail closed" {
  try completed_event(0, 10L, 11L, None, 12L, 1L, 1L) catch {
    @evidence.InvalidRequestEvent => ()
    _ => fail("unexpected generated-token shape error")
  } noraise {
    _ => fail("generated tokens without first-token time were admitted")
  }
  try
    @evidence.BenchmarkRequestEvent::new(
      request_ordinal=0,
      outcome=Rejected,
      submitted_at_ns=10L,
      admitted_at_ns=Some(11L),
      first_token_at_ns=None,
      terminal_at_ns=12L,
      input_tokens=1L,
      generated_tokens=0L,
    )
  catch {
    InvalidRequestEvent => ()
    _ => fail("unexpected rejected-event shape error")
  } noraise {
    _ => fail("admitted rejected request was accepted")
  }
}

///|
test "declared submissions require one terminal event each" {
  let events = [completed_event(0, 0L, 0L, None, 1L, 1L, 0L)]
  try
    @evidence.BenchmarkSummaryEvidence::admit(
      engine=LunaFlux,
      profile=Latency,
      ordinal=1,
      order_position=0,
      declared_submitted=2,
      measurement=test_digest('d'),
      events~,
    )
  catch {
    @evidence.IncorrectOutcomeAccounting => ()
    _ => fail("unexpected missing-outcome error")
  } noraise {
    _ => fail("missing terminal outcome was admitted")
  }
}

///|
test "request permutations and submission-time reversals are not canonical" {
  let permuted = [
    completed_event(1, 10L, 10L, None, 11L, 1L, 0L),
    completed_event(0, 11L, 11L, None, 12L, 1L, 0L),
  ]
  try admit_summary(permuted) catch {
    @evidence.NonCanonicalEventOrder => ()
    _ => fail("unexpected request-order error")
  } noraise {
    _ => fail("request permutation was silently canonicalized")
  }
  let reversed_time = [
    completed_event(0, 11L, 11L, None, 12L, 1L, 0L),
    completed_event(1, 10L, 10L, None, 13L, 1L, 0L),
  ]
  try admit_summary(reversed_time) catch {
    @evidence.NonCanonicalEventOrder => ()
    _ => fail("unexpected submission-order error")
  } noraise {
    _ => fail("decreasing submission time was admitted")
  }
}

///|
test "token and latency scalar overflow are rejected before hashing" {
  let token_overflow = [
    completed_event(0, 0L, 0L, None, 0L, 9223372036854775807L, 0L),
    completed_event(1, 0L, 0L, None, 0L, 1L, 0L),
  ]
  try admit_summary(token_overflow) catch {
    @evidence.SummaryIntegerOverflow => ()
    _ => fail("unexpected token-overflow error")
  } noraise {
    _ => fail("overflowing token total was admitted")
  }
  let latency_overflow = [
    completed_event(0, 0L, 0L, None, 9223372036854775807L, 0L, 0L),
    completed_event(1, 0L, 0L, None, 9223372036854775807L, 0L, 0L),
  ]
  try admit_summary(latency_overflow) catch {
    @evidence.SummaryIntegerOverflow => ()
    _ => fail("unexpected latency-overflow error")
  } noraise {
    _ => fail("overflowing latency total was admitted")
  }
}

///|
test "event arrays have a fixed admission bound" {
  let event = completed_event(0, 0L, 0L, None, 0L, 0L, 0L)
  let oversized = Array::make(
    @evidence.maximum_request_events_per_summary() + 1,
    event,
  )
  try
    @evidence.BenchmarkSummaryEvidence::admit(
      engine=LunaFlux,
      profile=Latency,
      ordinal=1,
      order_position=0,
      declared_submitted=oversized.length(),
      measurement=test_digest('d'),
      events=oversized,
    )
  catch {
    @evidence.BenchmarkEventLimitExceeded => ()
    _ => fail("unexpected event-limit error")
  } noraise {
    _ => fail("oversized event array was admitted")
  }
}

///|
test "legacy trial accounting cannot overflow Int before comparison" {
  try
    @evidence.BenchmarkTrial::new(
      engine=LunaFlux,
      profile=Latency,
      ordinal=1,
      order_position=0,
      submitted=1000000000,
      completed=1000000000,
      rejected=1000000000,
      timed_out=1000000000,
      cancelled=1000000000,
      failed=1000000000,
      correctness_passed=true,
      raw_events=test_digest('1'),
      correctness=test_digest('2'),
      summary=test_digest('3'),
    )
  catch {
    @evidence.IncorrectOutcomeAccounting => ()
    _ => fail("unexpected large-accounting error")
  } noraise {
    _ => fail("overflowing outcome accounting was admitted")
  }
}
