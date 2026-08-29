///|
priv struct BenchmarkEncoder {
  bytes : Array[Byte]
  maximum : Int
}

///|
fn BenchmarkEncoder::append_byte(
  self : BenchmarkEncoder,
  value : Byte,
) -> Unit raise BenchmarkEvidenceError {
  if self.bytes.length() >= self.maximum {
    raise CanonicalEvidenceTooLarge
  }
  self.bytes.push(value)
}

///|
fn BenchmarkEncoder::append_int(
  self : BenchmarkEncoder,
  value : Int,
) -> Unit raise BenchmarkEvidenceError {
  for byte in Int64::from_int(value).reinterpret_as_uint64().to_le_bytes() {
    self.append_byte(byte)
  }
}

///|
fn BenchmarkEncoder::append_int64(
  self : BenchmarkEncoder,
  value : Int64,
) -> Unit raise BenchmarkEvidenceError {
  for byte in value.reinterpret_as_uint64().to_le_bytes() {
    self.append_byte(byte)
  }
}

///|
fn BenchmarkEncoder::append_string(
  self : BenchmarkEncoder,
  value : StringView,
) -> Unit raise BenchmarkEvidenceError {
  let encoded = @utf8.encode(value)
  self.append_int(encoded.length())
  for byte in encoded {
    self.append_byte(byte)
  }
}

///|
fn BenchmarkEncoder::append_digest(
  self : BenchmarkEncoder,
  value : BenchmarkDigest,
) -> Unit raise BenchmarkEvidenceError {
  self.append_string(value.as_hex())
}

///|
fn encode_comparison(
  environment : BenchmarkEnvironment,
  engines : ArrayView[BenchmarkEngineIdentity],
  workloads : ArrayView[BenchmarkWorkload],
  trials : ArrayView[BenchmarkTrial],
) -> Bytes raise BenchmarkEvidenceError {
  let encoder : BenchmarkEncoder = { bytes: [], maximum: 1048576 }
  encoder.append_string("lunaflux.benchmark-comparison.v1")
  encoder.append_digest(environment.hardware)
  encoder.append_digest(environment.driver)
  encoder.append_digest(environment.toolkit)
  encoder.append_digest(environment.model)
  encoder.append_digest(environment.tokenizer)
  encoder.append_digest(environment.protocol)
  encoder.append_digest(environment.corpus)
  encoder.append_digest(environment.randomization)
  for identity in engines {
    encoder.append_int(engine_tag(identity.engine))
    encoder.append_digest(identity.revision)
    encoder.append_digest(identity.executable)
  }
  for workload in workloads {
    encoder.append_int(profile_tag(workload.profile))
    encoder.append_digest(workload.configuration)
  }
  for trial in trials {
    encoder.append_int(engine_tag(trial.engine))
    encoder.append_int(profile_tag(trial.profile))
    encoder.append_int(trial.ordinal)
    encoder.append_int(trial.order_position)
    encoder.append_int(trial.submitted)
    encoder.append_int(trial.completed)
    encoder.append_int(trial.rejected)
    encoder.append_int(trial.timed_out)
    encoder.append_int(trial.cancelled)
    encoder.append_int(trial.failed)
    encoder.append_int(if trial.correctness_passed { 1 } else { 0 })
    encoder.append_digest(trial.raw_events)
    encoder.append_digest(trial.correctness)
    encoder.append_digest(trial.summary)
  }
  Bytes::from_array(encoder.bytes)
}
