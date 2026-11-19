///|
test "dimension rounding uses nearest even at both kinds of tie" {
  assert_eq(ensure_divide(140.0, 56), 112)
  assert_eq(ensure_divide(196.0, 56), 224)
  assert_eq(ensure_divide(1.0, 56), 56)
}
