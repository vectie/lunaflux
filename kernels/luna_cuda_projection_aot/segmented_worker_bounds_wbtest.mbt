///|
test "segmented transfer guards specialize unsigned worker bounds" {
  for rows in [16, 32, 64] {
    for threads in [128, 256, 512] {
      let source = projection_sibling_phased_transfer_source(
        rows, 32, 128, 1024, threads,
      )
      assert_false(source.contains("threadIdx.x >= 0"))
      assert_false(source.contains("threadIdx.x < \{threads}"))
      assert_true(source.contains("transfer_active["))
      assert_true(source.contains("ld.global.v4.u32"))
      assert_true(source.contains("st.shared.v4.u32"))
    }
  }
}
