///|
fn BenchmarkTrialCollector::terminal_inputs(
  self : BenchmarkTrialCollector,
  request_ordinal : Int,
  outcome : @evidence.BenchmarkRequestOutcome,
  generated_tokens : Int64,
) -> (Int64, Int64?, Int64?, Int64) raise BenchmarkRunnerError {
  match self.slots[request_ordinal] {
    Vacant => raise BenchmarkRunnerInvalidLifecycle
    Terminal(_) => raise BenchmarkRunnerDuplicateRequest
    Submitted(submitted_at_ns, input_tokens) => {
      match outcome {
        Completed => raise BenchmarkRunnerInvalidLifecycle
        Rejected | TimedOut | Cancelled | Failed => ()
      }
      if generated_tokens != 0L {
        raise BenchmarkRunnerInvalidLifecycle
      }
      (submitted_at_ns, None, None, input_tokens)
    }
    Admitted(submitted_at_ns, admitted_at_ns, input_tokens) => {
      if outcome == Rejected || generated_tokens != 0L {
        raise BenchmarkRunnerInvalidLifecycle
      }
      (submitted_at_ns, Some(admitted_at_ns), None, input_tokens)
    }
    Streaming(submitted_at_ns, admitted_at_ns, first_token_at_ns, input_tokens) => {
      if outcome == Rejected || generated_tokens <= 0L {
        raise BenchmarkRunnerInvalidLifecycle
      }
      (
        submitted_at_ns,
        Some(admitted_at_ns),
        Some(first_token_at_ns),
        input_tokens,
      )
    }
  }
}

///|
fn BenchmarkTrialCollector::terminate_at(
  self : BenchmarkTrialCollector,
  request_ordinal : Int,
  outcome : @evidence.BenchmarkRequestOutcome,
  generated_tokens : Int64,
  timestamp_ns : Int64,
) -> Unit raise BenchmarkRunnerError {
  self.require_open()
  self.require_known_ordinal(request_ordinal)
  let (submitted_at_ns, admitted_at_ns, first_token_at_ns, input_tokens) = self.terminal_inputs(
    request_ordinal, outcome, generated_tokens,
  )
  let next_generated_tokens = checked_nonnegative_add(
    self.generated_tokens,
    generated_tokens,
  )
  self.validate_timestamp(timestamp_ns)
  let event = @evidence.BenchmarkRequestEvent::new(
    request_ordinal~,
    outcome~,
    submitted_at_ns~,
    admitted_at_ns~,
    first_token_at_ns~,
    terminal_at_ns=timestamp_ns,
    input_tokens~,
    generated_tokens~,
  ) catch {
    _ => raise BenchmarkRunnerEvidenceRejected
  }
  self.slots[request_ordinal] = Terminal(event)
  self.terminal = self.terminal + 1
  self.generated_tokens = next_generated_tokens
  self.last_timestamp_ns = Some(timestamp_ns)
}

///|
/// Records exactly one terminal outcome at the current monotonic sample.
/// Rejection is legal only before admission; a first-token observation requires
/// a positive terminal generated-token count.
pub fn BenchmarkTrialCollector::terminate(
  self : BenchmarkTrialCollector,
  request_ordinal~ : Int,
  outcome~ : @evidence.BenchmarkRequestOutcome,
  generated_tokens~ : Int64,
) -> Unit raise BenchmarkRunnerError {
  self.require_open()
  if self.captured_mode {
    raise BenchmarkRunnerInvalidLifecycle
  }
  self.terminate_at(
    request_ordinal,
    outcome,
    generated_tokens,
    self.read_timestamp_ns(),
  )
}

///|
pub fn BenchmarkTrialCollector::terminate_captured(
  self : BenchmarkTrialCollector,
  request_ordinal~ : Int,
  outcome~ : @evidence.BenchmarkRequestOutcome,
  generated_tokens~ : Int64,
  timestamp~ : CapturedMonotonicTimestamp,
) -> Unit raise BenchmarkRunnerError {
  self.require_open()
  if !self.captured_mode {
    raise BenchmarkRunnerInvalidLifecycle
  }
  self.terminate_at(
    request_ordinal,
    outcome,
    generated_tokens,
    timestamp.nanoseconds,
  )
}

///|
/// Finalizes only after every startup-declared request was submitted and
/// reached one terminal state. Terminal records are passed to the canonical
/// evidence package in request-ordinal order; this collector does not compute
/// or rewrite summaries itself.
pub fn BenchmarkTrialCollector::finish(
  self : BenchmarkTrialCollector,
) -> @evidence.BenchmarkSummaryEvidence raise BenchmarkRunnerError {
  self.require_open()
  if self.submitted != self.request_capacity || self.terminal != self.submitted {
    raise BenchmarkRunnerIncompleteTrial
  }
  let events : Array[@evidence.BenchmarkRequestEvent] = []
  for request_ordinal = 0
      request_ordinal < self.request_capacity
      request_ordinal = request_ordinal + 1 {
    match self.slots[request_ordinal] {
      Terminal(event) => events.push(event)
      _ => raise BenchmarkRunnerIncompleteTrial
    }
  }
  let measurements = @evidence.BenchmarkSummaryEvidence::admit(
    engine=self.engine,
    profile=self.profile,
    ordinal=self.ordinal,
    order_position=self.order_position,
    declared_submitted=self.request_capacity,
    measurement=self.measurement,
    events~,
  ) catch {
    _ => raise BenchmarkRunnerEvidenceRejected
  }
  self.finished = true
  measurements
}
