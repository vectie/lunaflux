///|
test "Qwen benchmark bridge uses the production framed receipt deadline" {
  assert_eq(bridge_request_deadline_millis(), 60000UL)
}

///|
test "Qwen benchmark bridge normalizes greedy seed to the internal contract" {
  let seed = try! @inference.SamplingSeed::new(bridge_internal_greedy_seed())
  assert_eq(seed.value(), 1UL)
}
