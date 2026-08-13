///|
test "system monotonic milliseconds never move backward" {
  let clock = MonotonicClock::system()
  let mut previous = clock.now_millis()
  for _ in 0..<1000 {
    let current = clock.now_millis()
    assert_true(current >= previous)
    previous = current
  }
}

///|
test "prepared monotonic read is one-shot and value requires success" {
  let prepared = MonotonicClock::system().prepare_read()
  try prepared.value() catch {
    Unavailable => ()
    _ => fail("unexpected unread prepared-clock error")
  } noraise {
    _ => fail("unread prepared clock exposed its default value")
  }
  assert_eq(prepared.read_status(), MonotonicReadReady)
  ignore(prepared.value())
  assert_eq(prepared.read_status(), MonotonicReadFailed)
  ignore(prepared.value())
}
