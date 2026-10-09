///|
extern "c" fn raw_test_event_wait() -> Int = "lunaflux_cuda_test_event_wait"

///|
test "cross stream event waits remain nonblocking and balance resource guards" {
  assert_eq(raw_test_event_wait(), 0)
}
