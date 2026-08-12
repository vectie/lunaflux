///|
test "probe is stable and bounded across repeated calls" {
  let first = @cuda.probe()
  for _ in 0..<128 {
    assert_eq(@cuda.probe(), first)
  }
  assert_true(first.device_count() >= 0)
  assert_true(first.driver_version() >= 0)
}
