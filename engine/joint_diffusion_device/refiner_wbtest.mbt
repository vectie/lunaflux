///|
test "refiner uses ungated pre-norm ordering with no modulation stage" {
  let stages = compose_refiner(0, [1, 2, 3, 4], 5, 6, [7, 8], 9)
  assert_eq(stages.length(), 10)
  for i, stage in stages {
    assert_eq(i, stage)
  }
}

///|
test "unrotated QK norm omits rotary operands instead of binding dummy tables" {
  let plain = normalization_arguments(0, 1, 2, 3, [])
  assert_eq(plain.length(), 6)
  assert_eq(plain[4], 0)
  assert_eq(plain[5], 1)
  let rotated = normalization_arguments(0, 1, 2, 3, [4, 5])
  assert_eq(rotated.length(), 8)
  assert_eq(rotated[4], 4)
  assert_eq(rotated[6], 0)
}
