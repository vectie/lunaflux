///|
extern "c" fn raw_test_execution_boundary(cycles : Int) -> Int = "lunaflux_cuda_test_execution_boundary"

///|
test "native transfers and GEMM retain balanced retryable ownership" {
  assert_eq(raw_test_execution_boundary(256), 0)
}

///|
test "native execution probe rejects unbounded repetition" {
  assert_eq(raw_test_execution_boundary(0), -2)
  assert_eq(raw_test_execution_boundary(10001), -2)
}
