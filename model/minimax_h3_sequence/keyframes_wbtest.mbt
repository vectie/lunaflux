///|
test "FL2VA reduction follows eight-lane grouping and bounded recursion" {
  let small = 5.0 / 3.0
  let large = 20.0 / 3.0
  let a = small + large
  let b = large + large
  let c = large + small
  let expected16 = a + b + (c + b) + (b + a + (b + c))
  assert_eq(keyframe_span(0, 16), expected16)
  for n in [1, 7, 8, 16, 37, 128, 129, 257] {
    let singles = (n + 4) / 5
    let analytic = singles.to_double() * small +
      (n - singles).to_double() * large
    assert_true((keyframe_span(0, n) - analytic).abs() < 0.00000000001)
  }
  assert_eq(keyframe_span(0, 129), keyframe_span(0, 64) + keyframe_span(64, 65))
}
