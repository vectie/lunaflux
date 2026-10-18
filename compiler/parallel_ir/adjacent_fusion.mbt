///|
/// Pure launch-coordinate rewrite. Replaces each adjacent pair by one launch;
/// effects and fork boundaries must not lie inside a removed pair.
pub struct AdjacentFusion {
  priv count : Int
  priv spans : ReadOnlyArray[(Int, Int)]
}

///|
pub fn AdjacentFusion::new(
  kernel_count~ : Int,
  pairs : ArrayView[Int],
) -> Self raise ParallelError {
  AdjacentFusion::spans(kernel_count~, pairs.map(first => (first, 2)))
}

///|
/// Fuse contiguous effect-free spans. Boundaries inside a removed span are
/// rejected rather than silently moved across a collective or fork edge.
pub fn AdjacentFusion::spans(
  kernel_count~ : Int,
  spans : ArrayView[(Int, Int)],
) -> Self raise ParallelError {
  guard kernel_count > 0 else { raise Capacity }
  let mut previous_end = 0
  for (first, length) in spans {
    guard first >= previous_end &&
      first >= 0 &&
      length >= 2 &&
      first <= kernel_count - length else {
      raise Dependency(first)
    }
    previous_end = first + length
  }
  { count: kernel_count, spans: ReadOnlyArray::from_array(spans), }
}

///|
pub fn AdjacentFusion::boundary(
  self : Self,
  old : Int,
) -> Int raise ParallelError {
  guard old >= 0 && old <= self.count else { raise Capacity }
  let mut removed = 0
  for (first, length) in self.spans {
    guard old <= first || old >= first + length else { raise Dependency(old) }
    if first + length <= old {
      removed += length - 1
    }
  }
  old - removed
}

///|
pub fn AdjacentFusion::fork(
  self : Self,
  span : ForkJoinSpan,
) -> ForkJoinSpan raise ParallelError {
  let (first, split, join) = span.range()
  ForkJoinSpan::new(
    first=self.boundary(first),
    split=self.boundary(split),
    join=self.boundary(join),
  )
}
