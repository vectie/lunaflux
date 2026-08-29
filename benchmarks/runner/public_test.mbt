///|
test "public collector samples the system monotonic millisecond domain" {
  assert_eq(
    system_monotonic_millis_measurement_digest().as_hex(),
    "db8b9983e81661e4c955ebaed2edf6c35ba789529fa88b2f02f8b7471ca2dd7b",
  )
  let collector = try! BenchmarkTrialCollector::new(
    engine=LunaFlux,
    profile=Latency,
    ordinal=1,
    order_position=0,
    request_capacity=1,
  )
  try! collector.submit(request_ordinal=0, input_tokens=1L)
  try! collector.admit(request_ordinal=0)
  try! collector.terminate(
    request_ordinal=0,
    outcome=Completed,
    generated_tokens=0L,
  )
  let measurements = try! collector.finish()
  assert_eq(measurements.summary().submitted(), 1)
  assert_eq(measurements.summary().completed(), 1)
  try collector.submit(request_ordinal=0, input_tokens=1L) catch {
    BenchmarkRunnerAlreadyFinished => ()
    _ => fail("unexpected post-finish failure")
  } noraise {
    _ => fail("finished collector was reopened")
  }
}
