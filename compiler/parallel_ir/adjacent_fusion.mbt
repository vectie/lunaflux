///|
/// Pure launch-coordinate rewrite. Replaces each adjacent pair by one launch;
/// effects and fork boundaries must not lie inside a removed pair.
pub struct AdjacentFusion {
  priv count : Int
  priv pairs : ReadOnlyArray[Int]
}

///|
pub fn AdjacentFusion::new(
  kernel_count~ : Int,
  pairs : ArrayView[Int],
) -> Self raise ParallelError {
  guard kernel_count > 0 else { raise Capacity }
  let mut previous = -2
  for first in pairs {
    guard first >= previous + 2 && first >= 0 && first < kernel_count - 1 else {
      raise Dependency(first)
    }
    previous = first
  }
  { count: kernel_count, pairs: ReadOnlyArray::from_array(pairs), }
}

///|
pub fn AdjacentFusion::boundary(
  self : Self,
  old : Int,
) -> Int raise ParallelError {
  guard old >= 0 && old <= self.count else { raise Capacity }
  let mut removed = 0
  for first in self.pairs {
    guard old != first + 1 else { raise Dependency(old) }
    if first + 2 <= old {
      removed += 1
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
