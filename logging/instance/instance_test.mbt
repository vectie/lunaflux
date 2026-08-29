///|
test "public instance log records and snapshots scalar events" {
  assert_eq(try! LunaInstanceLog::required_int_cells(2), 16UL)
  assert_eq(try! LunaInstanceLog::required_uint64_cells(2), 4UL)
  let log = try! LunaInstanceLog::new(2)
  assert_eq(log.capacity(), 2)
  try! log.record(
    LunaInstanceReady,
    LunaLogNoReason,
    timestamp_millis=10UL,
    count=1,
    duration_millis=2,
  )
  assert_true(try! (log.event_at(0) == LunaInstanceReady))
  assert_true(try! (log.reason_at(0) == LunaLogNoReason))
  assert_eq(try! log.timestamp_millis_at(0), 10UL)
  assert_eq(try! log.count_at(0), 1)
  assert_eq(try! log.duration_millis_at(0), 2)
  let snapshot = try! log.snapshot()
  assert_true(snapshot.is_live())
  assert_eq(try! snapshot.length(), 1)
  assert_true(try! (snapshot.event_at(0) == LunaInstanceReady))
}
