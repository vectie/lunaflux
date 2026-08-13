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
