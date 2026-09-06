///|
pub(all) enum FusionSchedule {
  Unfused
  ProducerSeparated
  FullyFused
} derive(Eq, Debug)

///|
pub(all) suberror FusionSelectionError {
  InvalidCost
  InvalidComparison
  NoFeasibleSchedule
} derive(Eq, Debug)

///|
priv enum CostBasis {
  Measured
  Estimated
} derive(Eq)

///|
pub struct FusionSpanCost {
  priv nanoseconds : Int64
  priv basis : CostBasis
} derive(Eq)

///|
pub fn FusionSpanCost::measured(
  median_nanoseconds~ : Int64,
  samples~ : Int,
) -> FusionSpanCost raise FusionSelectionError {
  if median_nanoseconds <= 0L || samples < 3 {
    raise InvalidCost
  }
  { nanoseconds: median_nanoseconds, basis: Measured }
}

///|
/// All terms cover the same complete semantic span. The backend supplies
/// effective compute/traffic costs including its own occupancy calibration.
pub fn FusionSpanCost::estimated(
  compute_nanoseconds~ : Int64,
  traffic_nanoseconds~ : Int64,
  synchronization_nanoseconds~ : Int64,
  launch_nanoseconds~ : Int64,
) -> FusionSpanCost raise FusionSelectionError {
  let mut total = 0L
  for
    term in [
      compute_nanoseconds, traffic_nanoseconds, synchronization_nanoseconds, launch_nanoseconds,
    ] {
    if term < 0L || total > 9223372036854775807L - term {
      raise InvalidCost
    }
    total += term
  }
  if total == 0L {
    raise InvalidCost
  }
  { nanoseconds: total, basis: Estimated }
}

///|
pub struct FusionAlternative {
  priv schedule : FusionSchedule
  priv cost : FusionSpanCost
  priv workspace_bytes : Int64
  priv local_bytes : Int64
}

///|
pub fn FusionAlternative::new(
  schedule~ : FusionSchedule,
  cost~ : FusionSpanCost,
  workspace_bytes~ : Int64,
  local_bytes~ : Int64,
) -> FusionAlternative raise FusionSelectionError {
  if workspace_bytes < 0L || local_bytes < 0L {
    raise InvalidCost
  }
  { schedule, cost, workspace_bytes, local_bytes }
}

///|
fn schedule_order(schedule : FusionSchedule) -> Int {
  match schedule {
    Unfused => 0
    ProducerSeparated => 1
    FullyFused => 2
  }
}

///|
/// Pure offline/startup choice. Never queries a device or times an operation.
pub fn select_fusion_schedule(
  alternatives : ArrayView[FusionAlternative],
  maximum_workspace_bytes~ : Int64,
  maximum_local_bytes~ : Int64,
) -> FusionSchedule raise FusionSelectionError {
  if alternatives.is_empty() ||
    alternatives.length() > 3 ||
    maximum_workspace_bytes < 0L ||
    maximum_local_bytes < 0L {
    raise InvalidComparison
  }
  let mut best : FusionAlternative? = None
  for index, candidate in alternatives {
    if candidate.cost.basis != alternatives[0].cost.basis {
      raise InvalidComparison
    }
    for prior in 0..<index {
      if candidate.schedule == alternatives[prior].schedule {
        raise InvalidComparison
      }
    }
    if candidate.workspace_bytes > maximum_workspace_bytes ||
      candidate.local_bytes > maximum_local_bytes {
      continue
    }
    match best {
      None => best = Some(candidate)
      Some(previous) =>
        if candidate.cost.nanoseconds < previous.cost.nanoseconds ||
          (
            candidate.cost.nanoseconds == previous.cost.nanoseconds &&
            schedule_order(candidate.schedule) <
            schedule_order(previous.schedule)
          ) {
          best = Some(candidate)
        }
    }
  }
  match best {
    Some(value) => value.schedule
    None => raise NoFeasibleSchedule
  }
}
