///|
test "subgroup placement preserves every descending tree level" {
  for workers in [32, 64, 128, 256, 512, 1024] {
    let (shared, local) = ReductionTreePartition::new(workers, 32).levels()
    let (baseline, empty) = ReductionTreePartition::new(workers, 1).levels()
    assert_eq(empty.length(), 0)
    assert_eq(
      shared.iter().to_array() + local.iter().to_array(),
      baseline.iter().to_array(),
    )
    assert_eq(local.iter().to_array(), [16, 8, 4, 2, 1])
  }
  for (workers, width) in [(256, 48), (16, 32)] {
    try ReductionTreePartition::new(workers, width) catch {
      _ => ()
    } noraise {
      _ => fail("illegal subgroup accepted")
    }
  }
}
