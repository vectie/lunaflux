///|
extern "c" fn raw_test_retryable_lifecycle(
  cycles : Int,
  failure_kind : Int,
) -> Int = "lunaflux_cuda_test_retryable_lifecycle"

///|
test "native cleanup remains retryable and retains parents" {
  for failure_kind in 1..<=2 {
    assert_eq(raw_test_retryable_lifecycle(512, failure_kind), 0)
  }
}

///|
test "native lifecycle probe bounds adversarial inputs" {
  assert_eq(raw_test_retryable_lifecycle(0, 1), -2)
  assert_eq(raw_test_retryable_lifecycle(1, 3), -2)
}
