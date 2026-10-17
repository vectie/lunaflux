///|
/// Host submission fences do not change the ordered device effect program.
pub(all) enum RetirementPolicy {
  EachCollective
  WholeFrame
} derive(Eq)

///|
pub extend RetirementPolicy with Eq::{equal, not_equal}

///|
pub(all) enum ReplayPolicy {
  Eager
  CaptureIntervals
  // Independent-row sums at capacity; active row count remains a device input.
  // Consumers must ignore inactive rows. Padding is not valid for all-gather.
  CapturePaddedCollectives
  // One-row decode and capacity-bounded prefill retain separate effect graphs.
  CaptureBucketedCollectives
  // Whole effect frame for one row; ordinary-count interval replay otherwise.
  CaptureDecodeCollectives
} derive(Eq)

///|
pub extend ReplayPolicy with Eq::{equal, not_equal}

///|
/// Immutable compute intervals separated by communication effects. A frame
/// owns every interval and collective until its terminal completion fence.
pub struct OrderedSubmissionPlan {
  priv ends : ReadOnlyArray[Int]
  priv retirement : RetirementPolicy
  priv replay : ReplayPolicy
}

///|
pub fn OrderedSubmissionPlan::new(
  kernel_count~ : Int,
  collective_cuts : ArrayView[Int],
  retirement~ : RetirementPolicy,
  replay? : ReplayPolicy = Eager,
) -> Self raise ParallelError {
  guard kernel_count > 0 && !collective_cuts.is_empty() else { raise Capacity }
  guard !(replay
    is (CapturePaddedCollectives
    | CaptureBucketedCollectives
    | CaptureDecodeCollectives)) ||
    retirement == WholeFrame else {
    raise Capacity
  }
  for i, cut in collective_cuts {
    guard cut > 0 &&
      cut < kernel_count &&
      (i == 0 || cut > collective_cuts[i - 1]) else {
      raise Step(i)
    }
  }
  let ends = collective_cuts.to_owned()
  ends.push(kernel_count)
  { ends: ReadOnlyArray::from_array(ends), retirement, replay, }
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
/// Scalar accessors avoid boxing a tuple in native submission loops.
pub fn OrderedSubmissionPlan::interval_start(self : Self, index : Int) -> Int {
  if index == 0 {
    0
  } else {
    self.ends[index - 1]
  }
}

///|
pub fn OrderedSubmissionPlan::interval_end(self : Self, index : Int) -> Int {
  self.ends[index]
}

///|
pub fn OrderedSubmissionPlan::retirement_policy(
  self : Self,
) -> RetirementPolicy {
  self.retirement
}

///|
pub fn OrderedSubmissionPlan::replay_policy(self : Self) -> ReplayPolicy {
  self.replay
}

///|
pub fn OrderedSubmissionPlan::collective_row_buckets(
  self : Self,
  capacity : Int,
) -> Array[Int] raise ParallelError {
  guard capacity > 0 else { raise Capacity }
  match self.replay {
    CaptureDecodeCollectives => [1]
    CapturePaddedCollectives => [capacity]
    CaptureBucketedCollectives =>
      if capacity == 1 {
        [1]
      } else {
        [1, capacity]
      }
    _ => []
  }
}

///|
pub fn OrderedSubmissionPlan::has_collective_graphs(self : Self) -> Bool {
  self.replay
  is (CapturePaddedCollectives
  | CaptureBucketedCollectives
  | CaptureDecodeCollectives)
}

///|
pub fn OrderedSubmissionPlan::captures_interval_graphs(self : Self) -> Bool {
  self.replay is (CaptureIntervals | CaptureDecodeCollectives)
}

///|
pub fn OrderedSubmissionPlan::replay_collectives(
  self : Self,
  live : Int,
) -> Bool {
  self.has_collective_graphs() &&
  (self.replay != CaptureDecodeCollectives || live == 1)
}

///|
pub fn OrderedSubmissionPlan::interval_queue_offset(self : Self) -> Int {
  if self.replay == CaptureDecodeCollectives {
    1
  } else {
    0
  }
}

///|
pub fn OrderedSubmissionPlan::collective_bucket(self : Self, live : Int) -> Int {
  if self.replay == CaptureBucketedCollectives && live > 1 {
    1
  } else {
    0
  }
}

///|
/// Caller-supplied backend estimates; no CUDA constants in this IR.
pub fn OrderedSubmissionPlan::replay_reserve_bytes(
  self : Self,
  bytes_per_kernel~ : Int64,
  bytes_per_interval~ : Int64,
) -> Int64 raise ParallelError {
  if self.replay == Eager {
    return 0L
  }
  guard bytes_per_kernel > 0L && bytes_per_interval > 0L else { raise Capacity }
  let kernels = self.ends[self.ends.length() - 1].to_int64()
  let intervals = self.ends.length().to_int64()
  guard bytes_per_kernel <= 9223372036854775807L / kernels &&
    bytes_per_interval <=
    (9223372036854775807L - kernels * bytes_per_kernel) / intervals else {
    raise Capacity
  }
  let reserve = kernels * bytes_per_kernel + intervals * bytes_per_interval
  if self.replay is (CaptureBucketedCollectives | CaptureDecodeCollectives) {
    guard reserve <= 4611686018427387903L else { raise Capacity }
    reserve * 2L
  } else {
    reserve
  }
}
