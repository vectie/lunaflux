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
test "framed connection pool event copy honors encoder work quantum" {
  assert_eq(luna_online_framed_pool_event_copy_amount(4096, 19, 64), 19)
  assert_eq(luna_online_framed_pool_event_copy_amount(4096, 19, 8), 8)
  assert_eq(luna_online_framed_pool_event_copy_amount(4, 19, 64), 4)
}

///|
test "local retirement backpressure never bypasses a framed writer" {
  assert_false(
    luna_online_framed_local_progress_preempts_writer(
      LunaOnlineFramedConnectionPoolBackpressured,
    ),
  )
  assert_true(
    luna_online_framed_local_progress_preempts_writer(
      LunaOnlineFramedConnectionPoolAdvanced,
    ),
  )
  assert_true(
    luna_online_framed_local_progress_preempts_writer(
      LunaOnlineFramedConnectionRetired,
    ),
  )
}

///|
test "framed event after transport cut discards without duplicate cancellation" {
  assert_false(
    luna_online_framed_pool_discard_after_transport_cut(
      LUNA_FRAMED_POOL_ACTIVE,
      false,
    ),
  )
  assert_true(
    luna_online_framed_pool_discard_after_transport_cut(
      LUNA_FRAMED_POOL_ACTIVE,
      true,
    ),
  )
  assert_true(
    luna_online_framed_pool_discard_after_transport_cut(
      LUNA_FRAMED_POOL_RETIRING,
      false,
    ),
  )
}

///|
test "framed connection pool rejection reasons preserve typed boundaries" {
  assert_true(
    luna_online_framed_pool_rejection_reason(LunaOnlineFramedRejectedDeadline) ==
    LunaLogDeadline,
  )
  assert_true(
    luna_online_framed_pool_rejection_reason(LunaOnlineFramedRejectedCapacity) ==
    LunaLogCapacity,
  )
  assert_true(
    luna_online_framed_pool_rejection_reason(LunaOnlineFramedRejectedService) ==
    LunaLogInternal,
  )
  assert_true(
    luna_online_framed_pool_rejection_reason(LunaOnlineFramedRejectedIdentity) ==
    LunaLogValidation,
  )
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
