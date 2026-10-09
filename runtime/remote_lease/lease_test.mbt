///|
fn lease() -> @remote_lease.RemoteLease raise @remote_lease.RemoteLeaseError {
  @remote_lease.RemoteLease::start(
    generation=7UL,
    duration_millis=100UL,
    local_now_millis=1000UL,
  )
}

///|
test "local time and strict heartbeat sequence determine an immutable deadline" {
  let initial = lease()
  let renewed = initial.renew(
    generation=7UL,
    sequence=1UL,
    local_now_millis=1050UL,
  )
  assert_eq(initial.deadline_millis(), 1100UL)
  assert_eq(initial.renewal_sequence_value(), 0UL)
  assert_eq(renewed.deadline_millis(), 1150UL)
  assert_eq(renewed.renewal_sequence_value(), 1UL)
  assert_eq(renewed.generation_value(), 7UL)
  assert_eq(renewed.poll(1149UL).status(), Active)
  assert_eq(renewed.poll(1150UL).status(), Expired)
}

///|
test "late renewal never revives a generation even without an earlier poll" {
  assert_eq(
    lease()
    .renew(generation=7UL, sequence=1UL, local_now_millis=1100UL)
    .status(),
    Expired,
  )
  let expired = lease().poll(1101UL)
  assert_eq(
    expired.renew(generation=8UL, sequence=1UL, local_now_millis=1102UL),
    expired,
  )
  assert_eq(expired.poll(900UL), expired)
  assert_eq(expired.transport_lost(), expired)
}

///|
test "replayed skipped and foreign heartbeats fence a live generation" {
  let initial = lease()
  let renewed = initial.renew(
    generation=7UL,
    sequence=1UL,
    local_now_millis=1050UL,
  )
  assert_eq(
    renewed
    .renew(generation=7UL, sequence=1UL, local_now_millis=1060UL)
    .status(),
    InvalidRenewal,
  )
  assert_eq(
    initial
    .renew(generation=7UL, sequence=0UL, local_now_millis=1050UL)
    .status(),
    InvalidRenewal,
  )
  assert_eq(
    initial
    .renew(generation=7UL, sequence=2UL, local_now_millis=1050UL)
    .status(),
    InvalidRenewal,
  )
  assert_eq(
    initial
    .renew(generation=8UL, sequence=1UL, local_now_millis=1050UL)
    .status(),
    InvalidRenewal,
  )
}

///|
test "clock regression and transport loss stay fenced" {
  let regressed = lease().poll(1050UL).poll(1049UL)
  assert_eq(regressed.status(), ClockRegressed)
  assert_eq(
    regressed.renew(generation=7UL, sequence=1UL, local_now_millis=1060UL),
    regressed,
  )
  let lost = lease().transport_lost()
  assert_eq(lost.status(), TransportLost)
  assert_eq(lost.poll(1001UL), lost)
  assert_eq(
    lost.renew(generation=7UL, sequence=1UL, local_now_millis=1001UL),
    lost,
  )
}

///|
test "deadline arithmetic cannot wrap into a live lease" {
  let start = @remote_lease.RemoteLease::start(
    generation=1UL,
    duration_millis=100UL,
    local_now_millis=0xffffffffffffffffUL - 100UL,
  )
  assert_eq(
    start
    .renew(
      generation=1UL,
      sequence=1UL,
      local_now_millis=0xffffffffffffffffUL - 50UL,
    )
    .status(),
    ClockOverflow,
  )
  try
    @remote_lease.RemoteLease::start(
      generation=1UL,
      duration_millis=100UL,
      local_now_millis=0xffffffffffffffffUL - 99UL,
    )
  catch {
    InvalidConfiguration => ()
  } noraise {
    _ => fail("overflow accepted")
  }
}
