///|
test "reusable clock supports repeated watchdog reads while one-shot remains distinct" {
  let clock = MonotonicClock::system().prepare_poller()
  try clock.value() catch {
    Unavailable => ()
    _ => fail("unexpected error")
  } noraise {
    _ => fail("unread clock accepted")
  }
  assert_true(clock.poll())
  let mut previous = clock.value()
  for _ in 0..<1000 {
    assert_true(clock.poll())
    let now = clock.value()
    assert_true(now >= previous)
    previous = now
  }
}
