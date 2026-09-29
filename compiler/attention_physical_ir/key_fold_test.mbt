///|
test "key fold keeps unrounded statistics and target-neutral ownership" {
  let fold = @attention_physical_ir.KeyFold::new(dimensions=129).unwrap()
  assert_eq(fold.scale_steps(), [MergeKeyStatistics, PublishKeyScales])
  assert_eq(fold.values_per_owner(16), Some(9))
  assert_eq(fold.values_per_owner(32), Some(5))
  assert_eq(fold.values_per_owner(64), Some(3))
  assert_eq(fold.values_per_owner(0), None)
  assert_eq(@attention_physical_ir.KeyFold::new(dimensions=0), None)
  assert_eq(
    @attention_physical_ir.KeyFold::new(dimensions=2147483647)
    .unwrap()
    .values_per_owner(2),
    Some(1073741824),
  )
}
