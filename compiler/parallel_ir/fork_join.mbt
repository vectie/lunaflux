///|
/// Two independent producer ranges, followed by a common consumer. The first
/// range uses the primary lane; the second is issued first on an auxiliary
/// lane. Serial lowering retains the original primary-then-auxiliary order.
/// This is an inert schedule, not an alias proof: the binding owner must retain
/// disjoint writable branch storage and shared read-only inputs until the join.
pub struct ForkJoinSpan {
  priv first : Int
  priv split : Int
  priv join : Int
}

///|
pub fn ForkJoinSpan::new(
  first~ : Int,
  split~ : Int,
  join~ : Int,
) -> Self raise ParallelError {
  guard first >= 0 && first < split && split < join else { raise Capacity }
  { first, split, join, }
}

///|
pub fn ForkJoinSpan::range(self : Self) -> (Int, Int, Int) {
  (self.first, self.split, self.join)
}

///|
pub fn ForkJoinSpan::shift(
  self : Self,
  offset : Int,
) -> Self raise ParallelError {
  guard offset >= -self.first && offset <= 2147483647 - self.join else {
    raise Capacity
  }
  {
    first: self.first + offset,
    split: self.split + offset,
    join: self.join + offset,
  }
}

///|
/// No nested forks and no collective may bisect a pair of producers. Both
/// branches must join before a reduction/publication boundary.
pub fn OrderedSubmissionPlan::validate_forks(
  self : Self,
  spans : ArrayView[ForkJoinSpan],
) -> Unit raise ParallelError {
  let mut previous = 0
  for span in spans {
    guard span.first >= previous &&
      span.join <= self.interval_end(self.interval_count() - 1) else {
      raise Capacity
    }
    for i in 0..<(self.interval_count() - 1) {
      let cut = self.interval_end(i)
      guard cut <= span.first || cut >= span.join else { raise Dependency(cut) }
    }
    previous = span.join
  }
}
