///|
pub(all) enum FusionSchedule {
  Unfused
  ProducerSeparated
  FullyFused
} derive(Eq, Debug)

///|
pub extend FusionSchedule with Eq::{equal, not_equal}

///|
pub extend FusionSchedule with @debug.Debug::{to_repr}

///|
pub(all) suberror FusionSelectionError {
  InvalidCost
} derive(Eq, Debug)

///|
pub extend FusionSelectionError with Eq::{equal, not_equal}

///|
pub extend FusionSelectionError with @debug.Debug::{to_repr}

///|
pub struct FusionSpanCost {
  priv region_cost : @fusion_regions.Cost
} derive(Eq)

///|
pub extend FusionSpanCost with Eq::{equal, not_equal}

///|
pub fn FusionSpanCost::measured(
  median_nanoseconds~ : Int64,
  samples~ : Int,
) -> FusionSpanCost raise FusionSelectionError {
  if median_nanoseconds <= 0L || samples < 3 {
    raise InvalidCost
  }
  {
    region_cost: @fusion_regions.Cost::measured(
      nanoseconds=median_nanoseconds,
      samples~,
    ) catch {
      _ => raise InvalidCost
    },
  }
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
  let total = for
    term in [
      compute_nanoseconds, traffic_nanoseconds, synchronization_nanoseconds, launch_nanoseconds,
    ]
    total = 0L {
    if term < 0L || total > 9223372036854775807L - term {
      raise InvalidCost
    }
    continue total + term
  } nobreak {
    total
  }
  if total == 0L {
    raise InvalidCost
  }
  {
    region_cost: @fusion_regions.Cost::estimated(nanoseconds=total) catch {
      _ => raise InvalidCost
    },
  }
}

///|
pub fn FusionSpanCost::region_cost(
  self : FusionSpanCost,
) -> @fusion_regions.Cost {
  self.region_cost
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
  { schedule, cost, workspace_bytes, local_bytes, }
}

///|
pub fn FusionAlternative::schedule(self : FusionAlternative) -> FusionSchedule {
  self.schedule
}

///|
pub fn FusionAlternative::cost(self : FusionAlternative) -> FusionSpanCost {
  self.cost
}

///|
pub fn FusionAlternative::resources(self : FusionAlternative) -> (Int64, Int64) {
  (self.workspace_bytes, self.local_bytes)
}

///|
