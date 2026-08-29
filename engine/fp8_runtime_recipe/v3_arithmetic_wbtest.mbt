///|
test "reusable v3 token-capacity comparison cannot overflow" {
  let maximum = 2147483647
  assert_true(fp8_reusable_token_capacity_contains(maximum, maximum, maximum))
  assert_true(fp8_reusable_token_capacity_contains(3, 7, 21))
  assert_false(fp8_reusable_token_capacity_contains(3, 7, 22))
  assert_false(fp8_reusable_token_capacity_contains(0, maximum, 1))
  assert_false(fp8_reusable_token_capacity_contains(maximum, 0, 1))
  assert_false(fp8_reusable_token_capacity_contains(maximum, maximum, 0))
}
