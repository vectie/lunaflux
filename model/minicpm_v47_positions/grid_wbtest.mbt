///|
test "FP32 linspace endpoints preserve upstream half-coordinate drift" {
  // Native ATen 2.9.1 CPU observations. Rational nearest-even gives 58,118,6.
  assert_eq(resampled_coordinate(7, 15, 118), 59)
  assert_eq(resampled_coordinate(7, 15, 236), 117)
  assert_eq(resampled_coordinate(11, 23, 14), 7)
  assert_eq(resampled_coordinate(0, 1, 118), 0)
  assert_eq(resampled_coordinate(0, 15, 118), 0)
  assert_eq(resampled_coordinate(14, 15, 118), 117)
}
