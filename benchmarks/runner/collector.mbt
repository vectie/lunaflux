///|
/// Creates one collector with an exact number of request lanes. Construction
/// allocates every lifecycle slot; observation never grows the slot owner.
pub fn BenchmarkTrialCollector::new(
  engine~ : @evidence.BenchmarkEngine,
  profile~ : @evidence.BenchmarkProfile,
  ordinal~ : Int,
  order_position~ : Int,
  request_capacity~ : Int,
) -> BenchmarkTrialCollector raise BenchmarkRunnerError {
  if request_capacity <= 0 ||
    request_capacity > @evidence.maximum_request_events_per_summary() {
    raise BenchmarkRunnerInvalidCapacity
  }
  if ordinal < 1 || ordinal > 3 || order_position < 0 || order_position > 2 {
    raise BenchmarkRunnerInvalidTrial
  }
  {
    engine,
    profile,
    ordinal,
    order_position,
    request_capacity,
    slots: FixedArray::make(request_capacity, Vacant),
    submitted: 0,
    terminal: 0,
    input_tokens: 0L,
    generated_tokens: 0L,
    last_timestamp_ns: None,
    finished: false,
    captured_mode: false,
    measurement: system_monotonic_millis_measurement_digest(),
    clock: @monotonic_clock.MonotonicClock::system(),
  }
}

///|
/// Creates a collector whose only legal public lifecycle ingress carries
/// externally captured monotonic timestamps. Its evidence is branded with a
/// distinct measurement digest and therefore cannot masquerade as live-clock
/// collection by this process.
pub fn BenchmarkTrialCollector::new_captured(
  engine~ : @evidence.BenchmarkEngine,
  profile~ : @evidence.BenchmarkProfile,
  ordinal~ : Int,
  order_position~ : Int,
  request_capacity~ : Int,
) -> BenchmarkTrialCollector raise BenchmarkRunnerError {
  let collector = BenchmarkTrialCollector::new(
    engine~,
    profile~,
    ordinal~,
    order_position~,
    request_capacity~,
  )
  {
    ..collector,
    captured_mode: true,
    measurement: captured_monotonic_nanos_measurement_digest(),
  }
}

///|
fn BenchmarkTrialCollector::require_open(
  self : BenchmarkTrialCollector,
) -> Unit raise BenchmarkRunnerError {
  if self.finished {
    raise BenchmarkRunnerAlreadyFinished
  }
}

///|
fn BenchmarkTrialCollector::require_known_ordinal(
  self : BenchmarkTrialCollector,
  request_ordinal : Int,
) -> Unit raise BenchmarkRunnerError {
  if request_ordinal < 0 || request_ordinal >= self.request_capacity {
    raise BenchmarkRunnerInvalidRequest
  }
}

///|
fn BenchmarkTrialCollector::submit_at(
  self : BenchmarkTrialCollector,
  request_ordinal : Int,
  input_tokens : Int64,
  timestamp_ns : Int64,
) -> Unit raise BenchmarkRunnerError {
  self.require_open()
  if self.submitted == self.request_capacity {
    raise BenchmarkRunnerCapacityExhausted
  }
  self.require_known_ordinal(request_ordinal)
  match self.slots[request_ordinal] {
    Vacant => ()
    _ => raise BenchmarkRunnerDuplicateRequest
  }
  if request_ordinal != self.submitted {
    raise BenchmarkRunnerNonCanonicalRequestOrder
  }
  let next_input_tokens = checked_nonnegative_add(
    self.input_tokens,
    input_tokens,
  )
  self.validate_timestamp(timestamp_ns)
  self.slots[request_ordinal] = Submitted(timestamp_ns, input_tokens)
  self.submitted = self.submitted + 1
  self.input_tokens = next_input_tokens
  self.last_timestamp_ns = Some(timestamp_ns)
}

///|
/// Records one canonical submission after sampling the owner-bound monotonic
/// clock. Request ordinals must be the exact sequence `0..<capacity`.
pub fn BenchmarkTrialCollector::submit(
  self : BenchmarkTrialCollector,
  request_ordinal~ : Int,
  input_tokens~ : Int64,
) -> Unit raise BenchmarkRunnerError {
  self.require_open()
  if self.captured_mode {
    raise BenchmarkRunnerInvalidLifecycle
  }
  self.submit_at(request_ordinal, input_tokens, self.read_timestamp_ns())
}

///|
/// Replays a submission timestamp sampled and preserved by an external live
/// runner. The collector still enforces canonical order and monotonicity.
pub fn BenchmarkTrialCollector::submit_captured(
  self : BenchmarkTrialCollector,
  request_ordinal~ : Int,
  input_tokens~ : Int64,
  timestamp~ : CapturedMonotonicTimestamp,
) -> Unit raise BenchmarkRunnerError {
  self.require_open()
  if !self.captured_mode {
    raise BenchmarkRunnerInvalidLifecycle
  }
  self.submit_at(request_ordinal, input_tokens, timestamp.nanoseconds)
}

///|
fn BenchmarkTrialCollector::admit_at(
  self : BenchmarkTrialCollector,
  request_ordinal : Int,
  timestamp_ns : Int64,
) -> Unit raise BenchmarkRunnerError {
  self.require_open()
  self.require_known_ordinal(request_ordinal)
  match self.slots[request_ordinal] {
    Submitted(submitted_at_ns, input_tokens) => {
      self.validate_timestamp(timestamp_ns)
      self.slots[request_ordinal] = Admitted(
        submitted_at_ns, timestamp_ns, input_tokens,
      )
      self.last_timestamp_ns = Some(timestamp_ns)
    }
    Terminal(_) => raise BenchmarkRunnerDuplicateRequest
    Vacant | Admitted(_, _, _) | Streaming(_, _, _, _) =>
      raise BenchmarkRunnerInvalidLifecycle
  }
}

///|
/// Records successful engine admission at the current monotonic sample.
pub fn BenchmarkTrialCollector::admit(
  self : BenchmarkTrialCollector,
  request_ordinal~ : Int,
) -> Unit raise BenchmarkRunnerError {
  self.require_open()
  if self.captured_mode {
    raise BenchmarkRunnerInvalidLifecycle
  }
  self.admit_at(request_ordinal, self.read_timestamp_ns())
}

///|
pub fn BenchmarkTrialCollector::admit_captured(
  self : BenchmarkTrialCollector,
  request_ordinal~ : Int,
  timestamp~ : CapturedMonotonicTimestamp,
) -> Unit raise BenchmarkRunnerError {
  self.require_open()
  if !self.captured_mode {
    raise BenchmarkRunnerInvalidLifecycle
  }
  self.admit_at(request_ordinal, timestamp.nanoseconds)
}

///|
fn BenchmarkTrialCollector::first_token_at(
  self : BenchmarkTrialCollector,
  request_ordinal : Int,
  timestamp_ns : Int64,
) -> Unit raise BenchmarkRunnerError {
  self.require_open()
  self.require_known_ordinal(request_ordinal)
  match self.slots[request_ordinal] {
    Admitted(submitted_at_ns, admitted_at_ns, input_tokens) => {
      self.validate_timestamp(timestamp_ns)
      self.slots[request_ordinal] = Streaming(
        submitted_at_ns, admitted_at_ns, timestamp_ns, input_tokens,
      )
      self.last_timestamp_ns = Some(timestamp_ns)
    }
    Terminal(_) => raise BenchmarkRunnerDuplicateRequest
    Vacant | Submitted(_, _) | Streaming(_, _, _, _) =>
      raise BenchmarkRunnerInvalidLifecycle
  }
}

///|
/// Records the first generated token at the current monotonic sample.
pub fn BenchmarkTrialCollector::first_token(
  self : BenchmarkTrialCollector,
  request_ordinal~ : Int,
) -> Unit raise BenchmarkRunnerError {
  self.require_open()
  if self.captured_mode {
    raise BenchmarkRunnerInvalidLifecycle
  }
  self.first_token_at(request_ordinal, self.read_timestamp_ns())
}

///|
pub fn BenchmarkTrialCollector::first_token_captured(
  self : BenchmarkTrialCollector,
  request_ordinal~ : Int,
  timestamp~ : CapturedMonotonicTimestamp,
) -> Unit raise BenchmarkRunnerError {
  self.require_open()
  if !self.captured_mode {
    raise BenchmarkRunnerInvalidLifecycle
  }
  self.first_token_at(request_ordinal, timestamp.nanoseconds)
}
