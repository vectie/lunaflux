///|
fn BenchmarkEncoder::append_optional_int64(
  self : BenchmarkEncoder,
  value : Int64?,
) -> Unit raise BenchmarkEvidenceError {
  match value {
    Some(inner) => {
      self.append_int(1)
      self.append_int64(inner)
    }
    None => self.append_int(0)
  }
}

///|
fn digest_benchmark_bytes(
  canonical : Bytes,
) -> BenchmarkDigest raise BenchmarkEvidenceError {
  BenchmarkDigest::from_sha256(
    @crypto.bytes_to_hex_string(@crypto.sha256(canonical)),
  )
}

///|
fn encode_raw_events(
  engine : BenchmarkEngine,
  profile : BenchmarkProfile,
  ordinal : Int,
  order_position : Int,
  measurement : BenchmarkDigest,
  events : ArrayView[BenchmarkRequestEvent],
) -> Bytes raise BenchmarkEvidenceError {
  let encoder : BenchmarkEncoder = { bytes: [], maximum: 1048576 }
  encoder.append_string("lunaflux.benchmark-request-events.v1")
  encoder.append_int(engine_tag(engine))
  encoder.append_int(profile_tag(profile))
  encoder.append_int(ordinal)
  encoder.append_int(order_position)
  encoder.append_digest(measurement)
  encoder.append_int(events.length())
  for event in events {
    encoder.append_int(event.request_ordinal)
    encoder.append_int(request_outcome_tag(event.outcome))
    encoder.append_int64(event.submitted_at_ns)
    encoder.append_optional_int64(event.admitted_at_ns)
    encoder.append_optional_int64(event.first_token_at_ns)
    encoder.append_int64(event.terminal_at_ns)
    encoder.append_int64(event.input_tokens)
    encoder.append_int64(event.generated_tokens)
  }
  Bytes::from_array(encoder.bytes)
}

///|
fn BenchmarkEncoder::append_integer_summary(
  self : BenchmarkEncoder,
  summary : BenchmarkIntegerSummary,
) -> Unit raise BenchmarkEvidenceError {
  self.append_int(summary.count)
  self.append_int64(summary.minimum)
  self.append_int64(summary.p50)
  self.append_int64(summary.p95)
  self.append_int64(summary.p99)
  self.append_int64(summary.maximum)
  self.append_int64(summary.total)
}

///|
fn encode_trial_summary(
  raw_events_digest : BenchmarkDigest,
  summary : BenchmarkTrialSummary,
) -> Bytes raise BenchmarkEvidenceError {
  let encoder : BenchmarkEncoder = { bytes: [], maximum: 1048576 }
  encoder.append_string("lunaflux.benchmark-trial-summary.v1")
  encoder.append_digest(raw_events_digest)
  encoder.append_int(summary.submitted)
  encoder.append_int(summary.completed)
  encoder.append_int(summary.rejected)
  encoder.append_int(summary.timed_out)
  encoder.append_int(summary.cancelled)
  encoder.append_int(summary.failed)
  encoder.append_int64(summary.input_tokens)
  encoder.append_int64(summary.generated_tokens)
  encoder.append_int64(summary.elapsed_ns)
  encoder.append_integer_summary(summary.queue_ns)
  encoder.append_integer_summary(summary.ttft_ns)
  encoder.append_integer_summary(summary.service_ns)
  encoder.append_integer_summary(summary.decode_ns)
  encoder.append_integer_summary(summary.end_to_end_ns)
  Bytes::from_array(encoder.bytes)
}
