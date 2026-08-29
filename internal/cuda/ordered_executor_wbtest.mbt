///|
extern "c" fn raw_test_ordered_executor(cycles : Int) -> Int = "lunaflux_cuda_test_ordered_executor"

///|
extern "c" fn raw_test_ordered_graph(cycles : Int) -> Int = "lunaflux_cuda_test_ordered_graph"

///|
test "ordered executor retains leases and completes by poll or abort" {
  assert_eq(raw_test_ordered_executor(1), 0)
  assert_eq(raw_test_ordered_executor(2), 0)
  assert_eq(raw_test_ordered_executor(3), 0)
  assert_eq(raw_test_ordered_executor(4), 0)
  assert_eq(raw_test_ordered_executor(5), 0)
  assert_eq(raw_test_ordered_executor(6), 0)
  assert_eq(raw_test_ordered_executor(256), 0)
}

///|
test "ordered executor probe rejects unsafe cycle bounds" {
  assert_eq(raw_test_ordered_executor(0), -2)
  assert_eq(raw_test_ordered_executor(10001), -2)
}

///|
test "ordered graph capture is immutable, optional, and retryably closed" {
  assert_eq(raw_test_ordered_graph(1), 0)
  assert_eq(raw_test_ordered_graph(2), 0)
  assert_eq(raw_test_ordered_graph(5), 0)
  assert_eq(raw_test_ordered_graph(8), 0)
  assert_eq(raw_test_ordered_graph(128), 0)
  assert_eq(raw_test_ordered_graph(0), -2)
  assert_eq(raw_test_ordered_graph(10001), -2)
}
