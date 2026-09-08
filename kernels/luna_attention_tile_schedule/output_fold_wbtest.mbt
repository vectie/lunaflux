///|
test "output fold residency is bounded by tile geometry not model identity" {
  for query in [16, 32] {
    for head in [16, 32, 64, 128] {
      assert_eq(
        output_fold_storage(AttentionMatrixQkPv, query, head, 8),
        SubgroupResidentOutputFold,
      )
    }
  }
  assert_eq(
    output_fold_storage(AttentionMatrixQkPv, 64, 128, 8),
    SharedOutputFold,
  )
  assert_eq(
    output_fold_storage(AttentionMatrixQkPv, 32, 256, 8),
    SharedOutputFold,
  )
  assert_eq(
    output_fold_storage(AttentionMatrixDot, 32, 128, 8),
    SharedOutputFold,
  )
  assert_eq(
    output_fold_storage(AttentionSubgroupDot, 1, 128, 8),
    SharedOutputFold,
  )
}
