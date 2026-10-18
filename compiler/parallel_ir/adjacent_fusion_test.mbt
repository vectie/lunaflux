///|
test "adjacent fusion remaps effects and rejects an interior effect" {
  let rewrite = AdjacentFusion::new(kernel_count=12, [2, 8])
  assert_eq(rewrite.boundary(2), 2)
  assert_eq(rewrite.boundary(4), 3)
  assert_eq(rewrite.boundary(12), 10)
  assert_eq(
    rewrite.fork(ForkJoinSpan::new(first=4, split=6, join=8)).range(),
    (3, 5, 7),
  )
  assert_true((try? rewrite.boundary(3)) is Err(_))
  assert_true((try? AdjacentFusion::new(kernel_count=12, [2, 3])) is Err(_))
  assert_true((try? AdjacentFusion::new(kernel_count=12, [11])) is Err(_))
}

///|
test "deferred three-stage fusion preserves collective and fork boundaries" {
  let rewrite = AdjacentFusion::spans(kernel_count=15, [(2, 3), (7, 2)])
  assert_eq(rewrite.boundary(2), 2)
  assert_eq(rewrite.boundary(5), 3)
  assert_eq(rewrite.boundary(15), 12)
  assert_eq(
    rewrite.fork(ForkJoinSpan::new(first=5, split=7, join=9)).range(),
    (3, 5, 6),
  )
  for boundary in [3, 4, 8] {
    assert_true((try? rewrite.boundary(boundary)) is Err(_))
  }
  for spans in [[(2, 3), (4, 2)], [(14, 2)], [(0, 1)]] {
    assert_true((try? AdjacentFusion::spans(kernel_count=15, spans)) is Err(_))
  }
}
