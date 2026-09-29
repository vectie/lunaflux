///|
test "segmented transfer guards specialize unsigned worker bounds" {
  for rows in [16, 32, 64] {
    for threads in [128, 256, 512] {
      let source = phased_transfer_fixture(
        projection_test_operand_storage(rows, 32, 128, threads),
        1024,
      )
      assert_false(source.contains("threadIdx.x >= 0"))
      assert_false(source.contains("threadIdx.x < \{threads}"))
      assert_true(source.contains("transfer_active["))
      assert_true(source.contains("ld.global.v4.u32"))
      assert_true(source.contains("st.shared.v4.u32"))
    }
  }
}
