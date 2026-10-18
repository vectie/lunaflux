///|
test "OpenAI pool preserves the original failure across cleanup errors" {
  let first = luna_openai_first_failure(None, "service", Failure("first"))
  let retained = luna_openai_first_failure(first, "cleanup", Failure("second"))
  guard retained is Some((stage, error)) else { fail("lost failure") }
  assert_eq(stage, "service")
  guard error is Failure(message) else { fail("changed error kind") }
  assert_eq(message, "first")
}
