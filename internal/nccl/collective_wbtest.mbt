///|
extern "c" fn raw_test_collectives(cycles : Int) -> Int = "lunaflux_nccl_test_collectives"

///|
test "fake native seam executes exact BF16 collectives and commits after poll" {
  @cuda.link_collective_interop()
  assert_eq(raw_test_collectives(256), 0)
}

///|
test "fake native collective seam rejects unbounded cycles" {
  @cuda.link_collective_interop()
  assert_eq(raw_test_collectives(0), -2)
  assert_eq(raw_test_collectives(100001), -2)
}
