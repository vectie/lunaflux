///|
test "deadline expiry classification preserves natural precedence" {
  assert_true(
    match classify_expiry_cut(OnlineDeadlinePending) {
      ExpiryCutPending => true
      _ => false
    },
  )
  assert_true(
    match classify_expiry_cut(OnlineDeadlineNaturalTerminal) {
      ExpiryCutNaturalTerminal => true
      _ => false
    },
  )
  assert_true(
    match classify_expiry_cut(OnlineDeadlineCommitted) {
      ExpiryCutCommitted => true
      _ => false
    },
  )
}
