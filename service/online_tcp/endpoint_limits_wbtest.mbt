///|
test "Luna online TCP bind validation rejects forged private limits" {
  let forged : LunaOnlineTcpLimits = {
    read_chunk_bytes: -1,
    write_chunk_bytes: 4096,
    accept_timeout_millis: 1000,
    input_idle_timeout_millis: 1000,
    write_timeout_millis: 1000,
    reactor_transition_budget: 17,
  }
  assert_false(forged.is_valid())
  let forged_budget : LunaOnlineTcpLimits = {
    read_chunk_bytes: 4096,
    write_chunk_bytes: 4096,
    accept_timeout_millis: 1000,
    input_idle_timeout_millis: 1000,
    write_timeout_millis: 1000,
    reactor_transition_budget: 65537,
  }
  assert_false(forged_budget.is_valid())

  let waits : LunaOnlineTcpLimits = {
    read_chunk_bytes: 4096,
    write_chunk_bytes: 4096,
    accept_timeout_millis: 5000,
    input_idle_timeout_millis: 1000,
    write_timeout_millis: 999,
    reactor_transition_budget: 17,
  }
  assert_true(waits.fits_transport_wait(1000))
  assert_false(waits.fits_transport_wait(999))
  assert_false(waits.fits_transport_wait(0))
  assert_eq(luna_online_tcp_safe_wait(1000, 0), 0)
  assert_eq(luna_online_tcp_safe_wait(1000, 1), 0)
  assert_eq(luna_online_tcp_safe_wait(1000, 2), 1)
  assert_eq(luna_online_tcp_safe_wait(17, 1000), 17)
  assert_eq(luna_online_tcp_safe_wait(1000, 17), 16)
}

///|
test "public transport-wait validator shares endpoint activation invariant" {
  let limits = try! LunaOnlineTcpLimits::new(
    read_chunk_bytes=64,
    write_chunk_bytes=64,
    accept_timeout_millis=1000,
    input_idle_timeout_millis=1000,
    write_timeout_millis=999,
    reactor_transition_budget=8,
  )
  try! limits.require_fits_transport_wait(1000)
  try limits.require_fits_transport_wait(999) catch {
    LunaOnlineTcpFailed(LunaOnlineTcpLimits) => ()
    _ => fail("unexpected transport wait coherence failure")
  } noraise {
    _ => fail("undersized transport wait was admitted")
  }
}
