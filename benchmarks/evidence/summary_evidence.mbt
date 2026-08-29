///|
/// Immutable canonical raw-event and summary artifacts for one trial. The
/// caller owns persistence; this value grants no runner or I/O authority.
pub struct BenchmarkSummaryEvidence {
  priv engine : BenchmarkEngine
  priv profile : BenchmarkProfile
  priv ordinal : Int
  priv order_position : Int
  priv raw_events_digest : BenchmarkDigest
  priv raw_events_canonical : Bytes
  priv summary_digest : BenchmarkDigest
  priv summary_canonical : Bytes
  priv summary : BenchmarkTrialSummary
}

///|
fn derive_trial_summary(
  declared_submitted : Int,
  events : ArrayView[BenchmarkRequestEvent],
) -> BenchmarkTrialSummary raise BenchmarkEvidenceError {
  if declared_submitted <= 0 || events.length() != declared_submitted {
    raise IncorrectOutcomeAccounting
  }
  if events.length() > BENCHMARK_MAX_REQUEST_EVENTS {
    raise BenchmarkEventLimitExceeded
  }
  let queue : Array[Int64] = []
  let ttft : Array[Int64] = []
  let service : Array[Int64] = []
  let decode : Array[Int64] = []
  let end_to_end : Array[Int64] = []
  let mut completed = 0
  let mut rejected = 0
  let mut timed_out = 0
  let mut cancelled = 0
  let mut failed = 0
  let mut input_tokens = 0L
  let mut generated_tokens = 0L
  let first_submitted = events[0].submitted_at_ns
  let mut previous_submitted = first_submitted
  let mut last_terminal = first_submitted
  for index, event in events {
    if event.request_ordinal != index ||
      (index > 0 && event.submitted_at_ns < previous_submitted) {
      raise NonCanonicalEventOrder
    }
    previous_submitted = event.submitted_at_ns
    last_terminal = Int64::max(last_terminal, event.terminal_at_ns)
    match event.outcome {
      Completed => completed = completed + 1
      Rejected => rejected = rejected + 1
      TimedOut => timed_out = timed_out + 1
      Cancelled => cancelled = cancelled + 1
      Failed => failed = failed + 1
    }
    input_tokens = checked_summary_add(input_tokens, event.input_tokens)
    generated_tokens = checked_summary_add(
      generated_tokens,
      event.generated_tokens,
    )
    end_to_end.push(event.terminal_at_ns - event.submitted_at_ns)
    match event.admitted_at_ns {
      Some(admitted) => {
        queue.push(admitted - event.submitted_at_ns)
        service.push(event.terminal_at_ns - admitted)
      }
      None => ()
    }
    match event.first_token_at_ns {
      Some(first_token) => {
        ttft.push(first_token - event.submitted_at_ns)
        decode.push(event.terminal_at_ns - first_token)
      }
      None => ()
    }
  }
  let accounted = completed + rejected + timed_out + cancelled + failed
  if accounted != declared_submitted {
    raise IncorrectOutcomeAccounting
  }
  {
    submitted: declared_submitted,
    completed,
    rejected,
    timed_out,
    cancelled,
    failed,
    input_tokens,
    generated_tokens,
    elapsed_ns: last_terminal - first_submitted,
    queue_ns: summarize_integers(queue),
    ttft_ns: summarize_integers(ttft),
    service_ns: summarize_integers(service),
    decode_ns: summarize_integers(decode),
    end_to_end_ns: summarize_integers(end_to_end),
  }
}

///|
/// Admits raw events and derives canonical integer summary evidence. Events
/// must be in request-ordinal order `0..<declared_submitted`, with
/// nondecreasing submission timestamps. The measurement digest identifies the
/// caller's clock/collection schema; this package does not execute it.
pub fn BenchmarkSummaryEvidence::admit(
  engine~ : BenchmarkEngine,
  profile~ : BenchmarkProfile,
  ordinal~ : Int,
  order_position~ : Int,
  declared_submitted~ : Int,
  measurement~ : BenchmarkDigest,
  events~ : ArrayView[BenchmarkRequestEvent],
) -> BenchmarkSummaryEvidence raise BenchmarkEvidenceError {
  if ordinal < 1 || ordinal > 3 || order_position < 0 || order_position > 2 {
    raise InvalidTrial
  }
  let summary = derive_trial_summary(declared_submitted, events)
  let raw_events_canonical = encode_raw_events(
    engine, profile, ordinal, order_position, measurement, events,
  )
  let raw_events_digest = digest_benchmark_bytes(raw_events_canonical)
  let summary_canonical = encode_trial_summary(raw_events_digest, summary)
  let summary_digest = digest_benchmark_bytes(summary_canonical)
  {
    engine,
    profile,
    ordinal,
    order_position,
    raw_events_digest,
    raw_events_canonical,
    summary_digest,
    summary_canonical,
    summary,
  }
}

///|
pub fn BenchmarkSummaryEvidence::raw_events_digest(
  self : BenchmarkSummaryEvidence,
) -> BenchmarkDigest {
  self.raw_events_digest
}

///|
pub fn BenchmarkSummaryEvidence::raw_events_canonical_bytes(
  self : BenchmarkSummaryEvidence,
) -> Bytes {
  self.raw_events_canonical
}

///|
pub fn BenchmarkSummaryEvidence::summary_digest(
  self : BenchmarkSummaryEvidence,
) -> BenchmarkDigest {
  self.summary_digest
}

///|
pub fn BenchmarkSummaryEvidence::summary_canonical_bytes(
  self : BenchmarkSummaryEvidence,
) -> Bytes {
  self.summary_canonical
}

///|
pub fn BenchmarkSummaryEvidence::summary(
  self : BenchmarkSummaryEvidence,
) -> BenchmarkTrialSummary {
  self.summary
}
