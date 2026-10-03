///|
test "split pair retention exactly covers strided component ownership" {
  for width in [64, 128, 256] {
    let plan = SplitPairRetention::new(width~, owners=32).unwrap()
    let (half, owners, pairs) = plan.geometry()
    let seen = Array::make(width, false)
    for owner in 0..<owners {
      for pair in 0..<pairs {
        let (first, second) = plan.components(owner, pair).unwrap()
        assert_eq(first % owners, owner)
        assert_eq(second % owners, owner)
        assert_eq(second - first, half)
        assert_false(seen[first])
        assert_false(seen[second])
        seen[first] = true
        seen[second] = true
      }
    }
    assert_true(seen.all(value => value))
    assert_eq(plan.components(-1, 0), None)
    assert_eq(plan.components(owners, 0), None)
    assert_eq(plan.components(0, pairs), None)
  }
}

///|
test "split pair retention rejects cross owner and invalid domains" {
  for (width, owners) in [(0, 32), (127, 32), (32, 32), (96, 32), (128, 0)] {
    assert_eq(SplitPairRetention::new(width~, owners~), None)
  }
}
