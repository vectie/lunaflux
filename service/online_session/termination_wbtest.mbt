///|
test "deadline expiry classification preserves natural precedence" {
  assert_eq(classify_expiry_cut(OnlineDeadlinePending), 0)
  assert_eq(classify_expiry_cut(OnlineDeadlineNaturalTerminal), 1)
  assert_eq(classify_expiry_cut(OnlineDeadlineCommitted), 2)
}
