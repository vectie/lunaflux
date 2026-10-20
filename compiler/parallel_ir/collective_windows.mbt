///|
/// Number of serial effect graphs in a logical row bucket. The supplied
/// horizon is a resource choice, not a model name or a vendor tuning constant.
pub fn OrderedSubmissionPlan::collective_window_count(
  self : Self,
  bucket : Int,
) -> Int {
  match self.replay {
    CaptureWindowedCollectives(_, sites) if bucket == 1 =>
      if self.interval_count() == 1 {
        1
      } else {
        (self.interval_count() - 2) / sites + 1
      }
    _ => 1
  }
}

///|
/// Half-open kernel and collective-site ranges. Windows cut only at existing
/// joined communication boundaries; they neither pad rows nor change sums.
pub fn OrderedSubmissionPlan::collective_window(
  self : Self,
  bucket : Int,
  window : Int,
) -> (Int, Int, Int, Int) raise ParallelError {
  guard bucket >= 0 &&
    bucket <= 1 &&
    window >= 0 &&
    window < self.collective_window_count(bucket) else {
    raise Capacity
  }
  let total = self.interval_count() - 1
  let (first, end) = match self.replay {
    CaptureWindowedCollectives(_, sites) if bucket == 1 => {
      let first = window * sites
      (first, first + sites.min(total - first))
    }
    _ => (0, total)
  }
  (
    if first == 0 {
      0
    } else {
      self.interval_end(first - 1)
    },
    if end == total {
      self.interval_end(total)
    } else {
      self.interval_end(end - 1)
    },
    first,
    end,
  )
}

///|
pub fn OrderedSubmissionPlan::collective_queue_first(
  self : Self,
  live : Int,
) -> Int {
  // The one-row bucket is always a single graph, even with bounded prefill.
  self.collective_bucket(live)
}

///|
test "bounded prefill preserves exact sites, tail, decode and fallback" {
  let p = OrderedSubmissionPlan::new(
    kernel_count=20,
    [3, 6, 9, 12, 15],
    retirement=WholeFrame,
    replay=CaptureWindowedCollectives(64, 2),
  )
  assert_eq(p.collective_window_count(0), 1)
  assert_eq(p.collective_window_count(1), 3)
  assert_eq(p.collective_window(1, 0), (0, 6, 0, 2))
  assert_eq(p.collective_window(1, 1), (6, 12, 2, 4))
  assert_eq(p.collective_window(1, 2), (12, 20, 4, 5))
  assert_eq(p.collective_window(0, 0), (0, 20, 0, 5))
  assert_eq(p.interval_queue_offset(), 4)
  assert_eq(p.collective_row_buckets(64), [1, 64])
  assert_true(p.replay_collectives(1))
  assert_true(p.replay_collectives(64))
  assert_false(p.replay_collectives(63))
  assert_true((try? p.collective_window(1, 3)) is Err(_))
}

///|
test "window horizons reject zero and preserve one-window limits" {
  for sites in [0, -1] {
    assert_true(
      (try? OrderedSubmissionPlan::new(
        kernel_count=8,
        [2, 5],
        retirement=WholeFrame,
        replay=CaptureWindowedCollectives(64, sites),
      ))
      is Err(_),
    )
  }
  let p = OrderedSubmissionPlan::new(
    kernel_count=8,
    [2, 5],
    retirement=WholeFrame,
    replay=CaptureWindowedCollectives(64, 2147483647),
  )
  assert_eq(p.collective_window_count(1), 1)
  assert_eq(p.collective_window(1, 0), (0, 8, 0, 2))
  assert_true((try? OrderedSubmissionPlan::new(
    kernel_count=8,
    [],
    retirement=WholeFrame,
    replay=CaptureWindowedCollectives(64, 1),
  )) is Err(_))
}
