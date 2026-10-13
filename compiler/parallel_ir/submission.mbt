///|
/// Host submission fences do not change the ordered device effect program.
pub(all) enum RetirementPolicy {
  EachCollective
  WholeFrame
} derive(Eq)

///|
pub extend RetirementPolicy with Eq::{equal, not_equal}

///|
/// Immutable compute intervals separated by communication effects. A frame
/// owns every interval and collective until its terminal completion fence.
pub struct OrderedSubmissionPlan {
  priv ends : ReadOnlyArray[Int]
  priv retirement : RetirementPolicy
}

///|
pub fn OrderedSubmissionPlan::new(
  kernel_count~ : Int,
  collective_cuts : ArrayView[Int],
  retirement~ : RetirementPolicy,
) -> Self raise ParallelError {
  guard kernel_count > 0 && !collective_cuts.is_empty() else { raise Capacity }
  for i, cut in collective_cuts {
    guard cut > 0 &&
      cut < kernel_count &&
      (i == 0 || cut > collective_cuts[i - 1]) else {
      raise Step(i)
    }
  }
  let ends = collective_cuts.to_owned()
  ends.push(kernel_count)
  { ends: ReadOnlyArray::from_array(ends), retirement, }
}

///|
pub fn OrderedSubmissionPlan::interval_count(self : Self) -> Int {
  self.ends.length()
}

///|
pub fn OrderedSubmissionPlan::interval(self : Self, index : Int) -> (Int, Int) {
  (if index == 0 { 0 } else { self.ends[index - 1] }, self.ends[index])
}

///|
pub fn OrderedSubmissionPlan::retirement_policy(
  self : Self,
) -> RetirementPolicy {
  self.retirement
}
