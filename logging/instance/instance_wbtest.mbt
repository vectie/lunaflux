///|
test "instance log finite event and reason vocabularies round-trip" {
  let events = [
    LunaInstanceStarting,
    LunaInstanceReady,
    LunaAdmissionAccepted,
    LunaAdmissionRejected,
    LunaRequestCompleted,
    LunaRequestCancelled,
    LunaRequestDeadline,
    LunaRequestFailed,
    LunaWorkerRestarted,
    LunaWorkerFailed,
    LunaNetworkAccepted,
    LunaNetworkDisconnected,
    LunaNetworkRejected,
    LunaBackpressureObserved,
    LunaDrainStarted,
    LunaInstanceClosed,
  ]
  for index in 0..<events.length() {
    assert_true(
      luna_instance_log_event_from_code(
        luna_instance_log_event_code(events[index]),
      ) ==
      events[index],
    )
  }
  let reasons = [
    LunaLogNoReason,
    LunaLogAuthentication,
    LunaLogValidation,
    LunaLogCapacity,
    LunaLogDeadline,
    LunaLogCancellation,
    LunaLogWorkerExit,
    LunaLogWorkerProtocol,
    LunaLogNetworkRead,
    LunaLogNetworkWrite,
    LunaLogPeerClosed,
    LunaLogServiceDraining,
    LunaLogInternal,
  ]
  for index in 0..<reasons.length() {
    assert_true(
      luna_instance_log_reason_from_code(
        luna_instance_log_reason_code(reasons[index]),
      ) ==
      reasons[index],
    )
  }
}

///|
test "instance log ring overwrites oldest and saturates drop count" {
  let log = try! LunaInstanceLog::new(3)
  for index in 0..<5 {
    try! log.record(
      LunaAdmissionAccepted,
      LunaLogNoReason,
      timestamp_millis=index.to_uint64(),
      count=index,
      duration_millis=index * 10,
    )
  }
  assert_eq(log.length(), 3)
  assert_eq(log.dropped_count(), 2UL)
  for index in 0..<3 {
    assert_eq(try! log.count_at(index), index + 2)
    assert_eq(try! log.timestamp_millis_at(index), (index + 2).to_uint64())
  }
  log.dropped_count = 0xffffffffffffffffUL
  try! log.record(
    LunaBackpressureObserved,
    LunaLogCapacity,
    timestamp_millis=5UL,
    count=1,
    duration_millis=0,
  )
  assert_eq(log.dropped_count(), 0xffffffffffffffffUL)
  assert_true(try! (log.event_at(2) == LunaBackpressureObserved))
}

///|
test "instance log rejects invalid scalars transactionally in frozen order" {
  let log = try! LunaInstanceLog::new(2)
  try! log.record(
    LunaInstanceStarting,
    LunaLogNoReason,
    timestamp_millis=10UL,
    count=0,
    duration_millis=0,
  )
  try
    log.record(
      LunaRequestFailed,
      LunaLogInternal,
      timestamp_millis=9UL,
      count=-1,
      duration_millis=-1,
    )
  catch {
    LunaInstanceLogInvalid(LunaInstanceLogTimestampRule) => ()
    _ => fail("unexpected instance log timestamp precedence")
  } noraise {
    _ => fail("backward instance log timestamp was accepted")
  }
  assert_eq(log.length(), 1)
  try
    log.record(
      LunaRequestFailed,
      LunaLogInternal,
      timestamp_millis=10UL,
      count=-1,
      duration_millis=-1,
    )
  catch {
    LunaInstanceLogInvalid(LunaInstanceLogCountRule) => ()
    _ => fail("unexpected instance log count precedence")
  } noraise {
    _ => fail("negative instance log count was accepted")
  }
  try
    log.record(
      LunaRequestFailed,
      LunaLogInternal,
      timestamp_millis=10UL,
      count=0,
      duration_millis=-1,
    )
  catch {
    LunaInstanceLogInvalid(LunaInstanceLogDurationRule) => ()
    _ => fail("unexpected instance log duration error")
  } noraise {
    _ => fail("negative instance log duration was accepted")
  }
  for invalid in [-1, LUNA_INSTANCE_LOG_MAX_COUNT + 1] {
    try
      log.record(
        LunaRequestFailed,
        LunaLogInternal,
        timestamp_millis=10UL,
        count=invalid,
        duration_millis=0,
      )
    catch {
      LunaInstanceLogInvalid(LunaInstanceLogCountRule) => ()
      _ => fail("unexpected bounded instance count error")
    } noraise {
      _ => fail("out-of-range instance count was accepted")
    }
  }
  for invalid in [-1, LUNA_INSTANCE_LOG_MAX_DURATION_MILLIS + 1] {
    try
      log.record(
        LunaRequestFailed,
        LunaLogInternal,
        timestamp_millis=10UL,
        count=0,
        duration_millis=invalid,
      )
    catch {
      LunaInstanceLogInvalid(LunaInstanceLogDurationRule) => ()
      _ => fail("unexpected bounded instance duration error")
    } noraise {
      _ => fail("out-of-range instance duration was accepted")
    }
  }
}

///|
test "instance log snapshot is stable and one-generation authenticated" {
  let log = try! LunaInstanceLog::new(2)
  try! log.record(
    LunaNetworkAccepted,
    LunaLogNoReason,
    timestamp_millis=20UL,
    count=1,
    duration_millis=0,
  )
  let first = try! log.snapshot()
  log.reset()
  try! log.record(
    LunaNetworkDisconnected,
    LunaLogPeerClosed,
    timestamp_millis=1UL,
    count=1,
    duration_millis=4,
  )
  assert_true(first.is_live())
  assert_true(try! (first.event_at(0) == LunaNetworkAccepted))
  assert_eq(try! first.timestamp_millis_at(0), 20UL)
  let second = try! log.snapshot()
  assert_false(first.is_live())
  assert_true(second.is_live())
  assert_true(try! (second.event_at(0) == LunaNetworkDisconnected))
  assert_true(try! (second.reason_at(0) == LunaLogPeerClosed))
  try first.event_at(-1) catch {
    LunaInstanceLogStaleSnapshot => ()
    _ => fail("instance log snapshot did not authenticate stale first")
  } noraise {
    _ => fail("stale instance log snapshot exposed an event")
  }
}

///|
test "instance log capacities bounds reset and snapshot exhaustion are exact" {
  for capacity in [-1, 0, LUNA_INSTANCE_LOG_MAX_CAPACITY + 1] {
    try LunaInstanceLog::new(capacity) catch {
      LunaInstanceLogInvalid(LunaInstanceLogCapacityRule) => ()
      _ => fail("unexpected instance log capacity error")
    } noraise {
      _ => fail("invalid instance log capacity was accepted")
    }
  }
  let maximum = try! LunaInstanceLog::new(LUNA_INSTANCE_LOG_MAX_CAPACITY)
  assert_eq(maximum.capacity(), LUNA_INSTANCE_LOG_MAX_CAPACITY)
  maximum.reset()

  let log = try! LunaInstanceLog::new(1)
  log.snapshot_generation = 0xffffffffffffffffUL
  try log.snapshot() catch {
    LunaInstanceLogSnapshotGenerationExhausted => ()
    _ => fail("unexpected instance log snapshot exhaustion error")
  } noraise {
    _ => fail("exhausted instance log snapshot generation was reused")
  }
  assert_eq(log.snapshot_length, 0)
}

///|
test "instance log allocation gate has a positive managed-storage control" {
  let positive = FixedArray::make(19, b'M')
  assert_eq(positive.length(), 19)
}
