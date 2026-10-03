///|
test "grouped matrix decode masks instruction rows without inventing tokens" {
  for heads in [1, 2, 4, 8, 16] {
    for keys in [32, 64] {
      for dimensions in [64, 128] {
        let plan = GroupedMatrixDecodeFold::new(heads~, keys~, dimensions~).unwrap()
        assert_eq(plan.fold().geometry(), (16, keys, dimensions))
        assert_eq(plan.numeric_law(), "grouped-head-matrix-bf16-probability-v1")
        assert_eq(plan.fold().exponential_law(), StrictNaturalExponential)
        for row in -1..<17 {
          assert_eq(plan.active_head(row), row >= 0 && row < heads)
        }
      }
    }
  }
  for
    (heads, keys, dimensions) in [
      (0, 32, 128),
      (17, 32, 128),
      (2, 48, 128),
      (2, 32, 80),
    ] {
    assert_eq(GroupedMatrixDecodeFold::new(heads~, keys~, dimensions~), None)
  }
}
