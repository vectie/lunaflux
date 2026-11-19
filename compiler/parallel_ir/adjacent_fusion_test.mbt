///|
test "adjacent fusion remaps effects and rejects an interior effect" {
  let rewrite = @parallel_ir.AdjacentFusion::new(kernel_count=12, [2, 8])
  assert_eq(rewrite.boundary(2), 2)
  assert_eq(rewrite.boundary(4), 3)
  assert_eq(rewrite.boundary(12), 10)
  assert_eq(
    rewrite
    .fork(@parallel_ir.ForkJoinSpan::new(first=4, split=6, join=8))
    .range(),
    (3, 5, 7),
  )
  try rewrite.boundary(3) catch {
    _ => ()
  } noraise {
    _ => fail("expected invalid input to fail")
  }
  try @parallel_ir.AdjacentFusion::new(kernel_count=12, [2, 3]) catch {
    _ => ()
  } noraise {
    _ => fail("expected invalid input to fail")
  }
  try @parallel_ir.AdjacentFusion::new(kernel_count=12, [11]) catch {
    _ => ()
  } noraise {
    _ => fail("expected invalid input to fail")
  }
}

///|
test "deferred three-stage fusion preserves collective and fork boundaries" {
  let rewrite = @parallel_ir.AdjacentFusion::spans(kernel_count=15, [
    (2, 3),
    (7, 2),
  ])
  assert_eq(rewrite.boundary(2), 2)
  assert_eq(rewrite.boundary(5), 3)
  assert_eq(rewrite.boundary(15), 12)
  assert_eq(
    rewrite
    .fork(@parallel_ir.ForkJoinSpan::new(first=5, split=7, join=9))
    .range(),
    (3, 5, 6),
  )
  for boundary in [3, 4, 8] {
    try rewrite.boundary(boundary) catch {
      _ => ()
    } noraise {
      _ => fail("expected invalid input to fail")
    }
  }
  for spans in [[(2, 3), (4, 2)], [(14, 2)], [(0, 1)]] {
    try @parallel_ir.AdjacentFusion::spans(kernel_count=15, spans) catch {
      _ => ()
    } noraise {
      _ => fail("expected invalid input to fail")
    }
  }
}
