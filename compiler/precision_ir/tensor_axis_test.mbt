///|
test "tensor axes are disjoint complete logical partitions" {
  for world in [2, 4, 8] {
    let mut covered = 0
    for rank in 0..<world {
      let (start, count) = TensorAxisPartition::new(64, rank~, world_size=world).range()
      assert_eq(start, covered)
      covered += count
    }
    assert_eq(covered, 64)
  }
  for
    (width, rank, world_size) in [
      (63, 0, 2),
      (64, 2, 2),
      (64, -1, 2),
      (64, 0, 1),
    ] {
    try TensorAxisPartition::new(width, rank~, world_size~) catch {
      IllegalPlacement => ()
      e => raise e
    } noraise {
      _ => fail("invalid partition accepted")
    }
  }
}

///|
test "dense partial plan budgets F32 output independently of BF16 product" {
  let plan = DenseSwiGluPrecision::new(
    hidden_width=64,
    intermediate_width=32,
    maximum_rows=3,
    publication=ReduceFloat32BeforeRound,
  )
  assert_eq(plan.product_bytes(), 192L)
  assert_eq(plan.workspace_bytes(), 960L)
  assert_true(plan.publication() == ReduceFloat32BeforeRound)
  ignore(plan.program())
}
