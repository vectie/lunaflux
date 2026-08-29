///|
/// One canonical terminal request record supplied by a benchmark harness.
/// Timestamps are monotonic-clock nanoseconds in one caller-declared domain;
/// this inert package never samples a clock itself.
pub struct BenchmarkRequestEvent {
  priv request_ordinal : Int
  priv outcome : BenchmarkRequestOutcome
  priv submitted_at_ns : Int64
  priv admitted_at_ns : Int64?
  priv first_token_at_ns : Int64?
  priv terminal_at_ns : Int64
  priv input_tokens : Int64
  priv generated_tokens : Int64
}

///|
/// Admits one internally monotonic request event. A positive generated-token
/// count requires a first-token timestamp; a rejected request cannot have
/// entered the engine or generated output.
pub fn BenchmarkRequestEvent::new(
  request_ordinal~ : Int,
  outcome~ : BenchmarkRequestOutcome,
  submitted_at_ns~ : Int64,
  admitted_at_ns~ : Int64?,
  first_token_at_ns~ : Int64?,
  terminal_at_ns~ : Int64,
  input_tokens~ : Int64,
  generated_tokens~ : Int64,
) -> BenchmarkRequestEvent raise BenchmarkEvidenceError {
  if request_ordinal < 0 ||
    submitted_at_ns < 0L ||
    terminal_at_ns < 0L ||
    input_tokens < 0L ||
    generated_tokens < 0L {
    raise InvalidRequestEvent
  }
  if terminal_at_ns < submitted_at_ns {
    raise InvalidEventTimestamp
  }
  match admitted_at_ns {
    Some(admitted) =>
      if admitted < submitted_at_ns || admitted > terminal_at_ns {
        raise InvalidEventTimestamp
      }
    None => ()
  }
  match first_token_at_ns {
    Some(first_token) =>
      match admitted_at_ns {
        Some(admitted) =>
          if first_token < admitted || first_token > terminal_at_ns {
            raise InvalidEventTimestamp
          }
        None => raise InvalidEventTimestamp
      }
    None => ()
  }
  if (generated_tokens > 0L) != (first_token_at_ns is Some(_)) {
    raise InvalidRequestEvent
  }
  match outcome {
    Completed => if admitted_at_ns is None { raise InvalidRequestEvent }
    Rejected =>
      if admitted_at_ns is Some(_) ||
        first_token_at_ns is Some(_) ||
        generated_tokens != 0L {
        raise InvalidRequestEvent
      }
    TimedOut | Cancelled | Failed => ()
  }
  {
    request_ordinal,
    outcome,
    submitted_at_ns,
    admitted_at_ns,
    first_token_at_ns,
    terminal_at_ns,
    input_tokens,
    generated_tokens,
  }
}

///|
pub fn BenchmarkRequestEvent::request_ordinal(
  self : BenchmarkRequestEvent,
) -> Int {
  self.request_ordinal
}

///|
pub fn BenchmarkRequestEvent::outcome(
  self : BenchmarkRequestEvent,
) -> BenchmarkRequestOutcome {
  self.outcome
}
