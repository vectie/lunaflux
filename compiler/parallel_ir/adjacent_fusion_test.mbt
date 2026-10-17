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
