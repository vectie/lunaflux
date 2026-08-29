///|
extern "c" fn raw_test_peer_access_boundary(cycles : Int) -> Int = "lunaflux_cuda_test_peer_access_boundary"

///|
test "directed peer access C ABI is exact and payload safe" {
  assert_eq(raw_test_peer_access_boundary(512), 0)
  assert_eq(raw_test_peer_access_boundary(0), -2)
  assert_eq(raw_test_peer_access_boundary(10001), -2)
}
