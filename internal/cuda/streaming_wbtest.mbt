///|
extern "c" fn streaming_probe(cycles : Int) -> Int = "lunaflux_cuda_test_streaming"

///|
test "streaming DMA retains leases through partial failures and retryable close" {
  assert_eq(streaming_probe(128), 0)
}
