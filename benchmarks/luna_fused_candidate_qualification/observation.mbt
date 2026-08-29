///|
pub(all) enum FusedCandidateBenchmarkPairOrder {
  BaselineThenCandidate
  CandidateThenBaseline
} derive(Debug, Eq)

///|
pub struct FusedCandidateBenchmarkArmObservation {
  priv duration_ns : Int64
  priv work_units : Int64
  priv resources_acquired : Int64
  priv resources_released : Int64
  priv live_resources_before : Int64
  priv live_resources_after : Int64
  priv live_bytes_before : Int64
  priv live_bytes_after : Int64
  priv resource_failures : Int64
  priv dispatch_canary_before : UInt
  priv dispatch_canary_after : UInt
}

///|
pub fn FusedCandidateBenchmarkArmObservation::new(
  duration_ns~ : Int64,
  work_units~ : Int64,
  resources_acquired~ : Int64,
  resources_released~ : Int64,
  live_resources_before~ : Int64,
  live_resources_after~ : Int64,
  live_bytes_before~ : Int64,
  live_bytes_after~ : Int64,
  resource_failures~ : Int64,
  dispatch_canary_before~ : UInt,
  dispatch_canary_after~ : UInt,
) -> FusedCandidateBenchmarkArmObservation raise FusedCandidateBenchmarkError {
  if duration_ns <= 0L ||
    work_units <= 0L ||
    resources_acquired < 0L ||
    resources_acquired > 1000000000L ||
    resources_released < 0L ||
    resources_released > 1000000000L ||
    live_resources_before < 0L ||
    live_resources_before > 1000000000L ||
    live_resources_after < 0L ||
    live_resources_after > 1000000000L ||
    live_bytes_before < 0L ||
    live_bytes_after < 0L ||
    resource_failures < 0L ||
    resource_failures > 1000000000L {
    raise InvalidFusedCandidateBenchmark(WorkCounter)
  }
  {
    duration_ns,
    work_units,
    resources_acquired,
    resources_released,
    live_resources_before,
    live_resources_after,
    live_bytes_before,
    live_bytes_after,
    resource_failures,
    dispatch_canary_before,
    dispatch_canary_after,
  }
}

///|
pub struct FusedCandidateBenchmarkPairedObservation {
  priv shape : FusedCandidateBenchmarkShape
  priv trial_ordinal : Int
  priv repetitions : Int
  priv order : FusedCandidateBenchmarkPairOrder
  priv baseline : FusedCandidateBenchmarkArmObservation
  priv candidate : FusedCandidateBenchmarkArmObservation
}

///|
pub fn FusedCandidateBenchmarkPairedObservation::new(
  shape~ : FusedCandidateBenchmarkShape,
  trial_ordinal~ : Int,
  repetitions~ : Int,
  order~ : FusedCandidateBenchmarkPairOrder,
  baseline~ : FusedCandidateBenchmarkArmObservation,
  candidate~ : FusedCandidateBenchmarkArmObservation,
) -> FusedCandidateBenchmarkPairedObservation raise FusedCandidateBenchmarkError {
  if trial_ordinal < 0 ||
    trial_ordinal > 1023 ||
    repetitions <= 0 ||
    repetitions > 1000000 {
    raise InvalidFusedCandidateBenchmark(ObservationOrder)
  }
  { shape, trial_ordinal, repetitions, order, baseline, candidate }
}
