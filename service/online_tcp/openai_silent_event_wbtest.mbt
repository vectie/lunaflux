///|
test "silent nonterminal rendering is acknowledged without closing SSE" {
  assert_true(luna_openai_silent_nonterminal(0, false))
  assert_false(luna_openai_silent_nonterminal(0, true))
  assert_false(luna_openai_silent_nonterminal(8, false))
  assert_false(luna_openai_silent_nonterminal(8, true))
  assert_false(luna_openai_silent_nonterminal(-1, false))
}
