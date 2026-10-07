///|
test "fewer registers without higher residency cannot outrank faster completion" {
  let s = scope(shape="decode-r2-e8-h32767")
  let hardware = @resource_policy.ResourceBudget::new(
    max_threads=128,
    max_shared_bytes=65536L,
    max_registers_per_thread=Some(255),
    min_resident_groups=Some(2),
  ).unwrap()
  let options = [
    candidate("unrolled", shared=33040L, registers=Some(148)),
    candidate("windowed", shared=33040L, registers=Some(141)),
  ]
  let measured = [
    observation(s, "unrolled", 1450717L, registers=148),
    observation(s, "windowed", 2025762L, registers=141),
  ]
  for preferred in ["unrolled", "windowed"] {
    let result = @resource_policy.select(
      s,
      hardware,
      options,
      measured,
      preferred_id=preferred,
    ).unwrap()
    assert_eq(result.policy().id(), "unrolled")
    assert_eq(
      result.origin(),
      @resource_policy.Measured(latency_ns=1450717L, samples=5),
    )
  }
}
