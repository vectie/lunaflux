///|
test "framed connection pool limits are explicit and bounded" {
  let limits = LunaOnlineFramedConnectionPoolLimits::new(
    connection_capacity=2,
    io_poll_timeout_millis=1,
    request_work_units=17,
    event_work_units=19,
  )
  assert_eq(limits.connection_capacity(), 2)
  assert_eq(limits.io_poll_timeout_millis(), 1)
  for
    make in [
      () => {
        LunaOnlineFramedConnectionPoolLimits::new(
          connection_capacity=0,
          io_poll_timeout_millis=1,
          request_work_units=1,
          event_work_units=1,
        )
      },
      () => {
        LunaOnlineFramedConnectionPoolLimits::new(
          connection_capacity=2,
          io_poll_timeout_millis=0,
          request_work_units=1,
          event_work_units=1,
        )
      },
      () => {
        LunaOnlineFramedConnectionPoolLimits::new(
          connection_capacity=2,
          io_poll_timeout_millis=1,
          request_work_units=0,
          event_work_units=1,
        )
      },
      () => {
        LunaOnlineFramedConnectionPoolLimits::new(
          connection_capacity=2,
          io_poll_timeout_millis=1,
          request_work_units=1,
          event_work_units=0,
        )
      },
    ] {
    try make() catch {
      LunaOnlineTcpFailed(LunaOnlineTcpLimits) => ()
      _ => fail("unexpected framed-pool limit error")
    } noraise {
      _ => fail("invalid framed-pool limits were accepted")
    }
  }
}

///|
test "framed connection pool readiness is fail closed" {
  let phase_ready = fn(
    closed : Bool,
    failed : Bool,
    draining : Bool,
    listener_count : Int,
    service_count : Int,
    stream_count : Int,
    stream_live : Bool,
  ) -> Bool {
    !closed &&
    !failed &&
    !draining &&
    listener_count == 1 &&
    service_count == 1 &&
    stream_count == 1 &&
    stream_live
  }
  assert_true(phase_ready(false, false, false, 1, 1, 1, true))
  assert_false(phase_ready(false, false, true, 1, 1, 1, true))
  assert_false(phase_ready(false, true, false, 1, 1, 1, true))
  assert_false(phase_ready(false, false, false, 0, 1, 1, true))
  assert_false(phase_ready(false, false, false, 1, 1, 1, false))
}
