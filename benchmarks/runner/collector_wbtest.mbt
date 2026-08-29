///|
fn runner_collector(capacity? : Int = 2) -> BenchmarkTrialCollector {
  try! BenchmarkTrialCollector::new(
    engine=LunaFlux,
    profile=Latency,
    ordinal=1,
    order_position=0,
    request_capacity=capacity,
  )
}

///|
fn record_completed(
  collector : BenchmarkTrialCollector,
  request_ordinal : Int,
  start_ns : Int64,
  input_tokens? : Int64 = 8L,
  generated_tokens? : Int64 = 2L,
) -> Unit raise BenchmarkRunnerError {
  collector.submit_at(request_ordinal, input_tokens, start_ns)
  collector.admit_at(request_ordinal, start_ns + 1L)
  collector.first_token_at(request_ordinal, start_ns + 2L)
  collector.terminate_at(
    request_ordinal,
    Completed,
    generated_tokens,
    start_ns + 3L,
  )
}

///|
test "collector constructs exact canonical trial evidence" {
  let collector = runner_collector()
  try! record_completed(collector, 0, 10L)
  try! collector.submit_at(1, 9L, 20L)
  try! collector.terminate_at(1, Rejected, 0L, 21L)
  let evidence = try! collector.finish()
  let summary = evidence.summary()
  assert_eq(summary.submitted(), 2)
  assert_eq(summary.completed(), 1)
  assert_eq(summary.rejected(), 1)
  assert_eq(summary.input_tokens(), 17L)
  assert_eq(summary.generated_tokens(), 2L)
  assert_eq(summary.elapsed_ns(), 11L)
  assert_eq(summary.queue_ns().total(), 1L)
  assert_eq(summary.ttft_ns().total(), 2L)
  assert_eq(summary.service_ns().total(), 2L)
  assert_eq(summary.decode_ns().total(), 1L)
  assert_eq(summary.end_to_end_ns().total(), 4L)
}

///|
test "clock reversal is rejected without advancing lifecycle" {
  let collector = runner_collector(capacity=1)
  try! collector.submit_at(0, 1L, 10L)
  try collector.admit_at(0, 9L) catch {
    BenchmarkRunnerClockMovedBackward => ()
    _ => fail("unexpected reordered-clock failure")
  } noraise {
    _ => fail("reordered clock was admitted")
  }
  try! collector.admit_at(0, 10L)
  try! collector.terminate_at(0, Completed, 0L, 11L)
  ignore(try! collector.finish())
}

///|
test "every startup lane requires one terminal outcome" {
  let admitted = runner_collector(capacity=1)
  try! admitted.submit_at(0, 1L, 1L)
  try! admitted.admit_at(0, 2L)
  try admitted.finish() catch {
    BenchmarkRunnerIncompleteTrial => ()
    _ => fail("unexpected missing-terminal failure")
  } noraise {
    _ => fail("admitted request without a terminal outcome was finalized")
  }
  let missing_submission = runner_collector(capacity=2)
  try! missing_submission.submit_at(0, 1L, 1L)
  try! missing_submission.terminate_at(0, Rejected, 0L, 2L)
  try missing_submission.finish() catch {
    BenchmarkRunnerIncompleteTrial => ()
    _ => fail("unexpected missing-submission failure")
  } noraise {
    _ => fail("trial with an unused startup lane was finalized")
  }
}

///|
test "duplicate request and terminal observations fail closed" {
  let collector = runner_collector(capacity=2)
  try! collector.submit_at(0, 1L, 1L)
  try collector.submit_at(0, 1L, 2L) catch {
    BenchmarkRunnerDuplicateRequest => ()
    _ => fail("unexpected duplicate-submission failure")
  } noraise {
    _ => fail("duplicate submission was admitted")
  }
  try! collector.terminate_at(0, Rejected, 0L, 2L)
  try collector.terminate_at(0, Rejected, 0L, 3L) catch {
    BenchmarkRunnerDuplicateRequest => ()
    _ => fail("unexpected duplicate-terminal failure")
  } noraise {
    _ => fail("duplicate terminal outcome was admitted")
  }
}

///|
test "request submission order is canonical and not repaired" {
  let collector = runner_collector(capacity=2)
  try collector.submit_at(1, 1L, 1L) catch {
    BenchmarkRunnerNonCanonicalRequestOrder => ()
    _ => fail("unexpected request-order failure")
  } noraise {
    _ => fail("out-of-order request was admitted")
  }
  try! collector.submit_at(0, 1L, 1L)
}

///|
test "capacity exhaustion is distinct and deterministic" {
  let collector = runner_collector(capacity=1)
  try! collector.submit_at(0, 1L, 1L)
  try collector.submit_at(1, 1L, 2L) catch {
    BenchmarkRunnerCapacityExhausted => ()
    _ => fail("unexpected capacity failure")
  } noraise {
    _ => fail("request beyond startup capacity was admitted")
  }
}

///|
test "checked token accounting rejects overflow transactionally" {
  let collector = runner_collector(capacity=2)
  try! collector.submit_at(0, 9223372036854775807L, 1L)
  try collector.submit_at(1, 1L, 2L) catch {
    BenchmarkRunnerIntegerOverflow => ()
    _ => fail("unexpected token-overflow failure")
  } noraise {
    _ => fail("overflowing input-token total was admitted")
  }
  try! collector.submit_at(1, 0L, 2L)
  try! collector.terminate_at(0, Rejected, 0L, 3L)
  try! collector.terminate_at(1, Rejected, 0L, 4L)
  let evidence = try! collector.finish()
  assert_eq(evidence.summary().input_tokens(), 9223372036854775807L)
}

///|
test "generated-token total also uses checked arithmetic" {
  let collector = runner_collector(capacity=2)
  try! collector.submit_at(0, 0L, 1L)
  try! collector.admit_at(0, 2L)
  try! collector.first_token_at(0, 3L)
  try! collector.terminate_at(0, Completed, 9223372036854775806L, 4L)
  try! collector.submit_at(1, 0L, 5L)
  try! collector.admit_at(1, 6L)
  try! collector.first_token_at(1, 7L)
  try collector.terminate_at(1, Completed, 2L, 8L) catch {
    BenchmarkRunnerIntegerOverflow => ()
    _ => fail("unexpected generated-token overflow failure")
  } noraise {
    _ => fail("overflowing generated-token total was admitted")
  }
  try! collector.terminate_at(1, Completed, 1L, 8L)
  let evidence = try! collector.finish()
  assert_eq(evidence.summary().generated_tokens(), 9223372036854775807L)
}

///|
fn deterministic_trial() -> @evidence.BenchmarkSummaryEvidence {
  let collector = runner_collector()
  try! record_completed(collector, 0, 100L)
  try! collector.submit_at(1, 3L, 110L)
  try! collector.admit_at(1, 111L)
  try! collector.terminate_at(1, Cancelled, 0L, 112L)
  try! collector.finish()
}

///|
test "identical lifecycle observations produce identical evidence" {
  let first = deterministic_trial()
  let second = deterministic_trial()
  assert_eq(
    first.raw_events_digest().as_hex(),
    second.raw_events_digest().as_hex(),
  )
  assert_eq(first.summary_digest().as_hex(), second.summary_digest().as_hex())
  assert_eq(
    first.raw_events_canonical_bytes(),
    second.raw_events_canonical_bytes(),
  )
  assert_eq(first.summary_canonical_bytes(), second.summary_canonical_bytes())
}

///|
test "captured collector is clock-branded and live ingress cannot cross modes" {
  assert_not_eq(
    captured_monotonic_nanos_measurement_digest().as_hex(),
    system_monotonic_millis_measurement_digest().as_hex(),
  )
  let captured = try! BenchmarkTrialCollector::new_captured(
    engine=LunaFlux,
    profile=Latency,
    ordinal=1,
    order_position=0,
    request_capacity=1,
  )
  try captured.submit(request_ordinal=0, input_tokens=1L) catch {
    BenchmarkRunnerInvalidLifecycle => ()
    _ => fail("unexpected captured/live mode rejection")
  } noraise {
    _ => fail("captured collector accepted live-clock ingress")
  }
  let t0 = try! CapturedMonotonicTimestamp::from_nanoseconds(10L)
  let t1 = try! CapturedMonotonicTimestamp::from_nanoseconds(11L)
  captured.submit_captured(request_ordinal=0, input_tokens=1L, timestamp=t0)
  captured.terminate_captured(
    request_ordinal=0,
    outcome=Rejected,
    generated_tokens=0L,
    timestamp=t1,
  )
  ignore(captured.finish())
  let live = runner_collector(capacity=1)
  try
    live.submit_captured(request_ordinal=0, input_tokens=1L, timestamp=t0)
  catch {
    BenchmarkRunnerInvalidLifecycle => ()
    _ => fail("unexpected live/captured mode rejection")
  } noraise {
    _ => fail("live collector accepted captured-clock ingress")
  }
}
