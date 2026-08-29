///|
/// Fail-closed stages for the physical LunaFlux benchmark adapter.
pub enum RuntimeSpawnedBenchmarkRule {
  SpawnedBenchmarkContract
  SpawnedBenchmarkCollector
  SpawnedBenchmarkServing
  SpawnedBenchmarkCleanup
} derive(Debug, Eq)

///|
pub fn RuntimeSpawnedBenchmarkRule::code(
  self : RuntimeSpawnedBenchmarkRule,
) -> String {
  match self {
    SpawnedBenchmarkContract => "contract"
    SpawnedBenchmarkCollector => "collector"
    SpawnedBenchmarkServing => "serving"
    SpawnedBenchmarkCleanup => "cleanup"
  }
}

///|
pub suberror RuntimeSpawnedBenchmarkError {
  RuntimeSpawnedBenchmarkFailed(RuntimeSpawnedBenchmarkRule)
} derive(Debug, Eq)

///|
fn spawned_benchmark_error(
  rule : RuntimeSpawnedBenchmarkRule,
) -> RuntimeSpawnedBenchmarkError {
  RuntimeSpawnedBenchmarkFailed(rule)
}

///|
/// Immutable single-request trial declaration. The engine is deliberately
/// absent: this adapter always records `LunaFlux`. The enclosed serving
/// contract already owns an authenticated copy of the canonical request.
pub struct RuntimeSpawnedBenchmarkContract {
  priv serving : RuntimeSpawnedServingValidationContract
  priv profile : @benchmark_evidence.BenchmarkProfile
  priv ordinal : Int
  priv order_position : Int
  priv verify_expected_tokens : Bool
}

///|
/// Canonical runner evidence paired with the terminal serving proof that was
/// observed on the same singular runtime owner.
pub struct RuntimeSpawnedBenchmarkResult {
  priv measurements : @benchmark_evidence.BenchmarkSummaryEvidence
  priv serving : RuntimeSpawnedServingValidationResult
  priv startup : RuntimeStartupMeasurement
}

///|
pub fn RuntimeSpawnedBenchmarkContract::new(
  serving : RuntimeSpawnedServingValidationContract,
  profile : @benchmark_evidence.BenchmarkProfile,
  ordinal : Int,
  order_position : Int,
) -> RuntimeSpawnedBenchmarkContract raise RuntimeSpawnedBenchmarkError {
  if ordinal < 1 || ordinal > 3 || order_position < 0 || order_position > 2 {
    raise spawned_benchmark_error(SpawnedBenchmarkContract)
  }
  { serving, profile, ordinal, order_position, verify_expected_tokens: true }
}

///|
pub fn RuntimeSpawnedBenchmarkResult::measurements(
  self : RuntimeSpawnedBenchmarkResult,
) -> @benchmark_evidence.BenchmarkSummaryEvidence {
  self.measurements
}

///|
pub fn RuntimeSpawnedBenchmarkResult::serving(
  self : RuntimeSpawnedBenchmarkResult,
) -> RuntimeSpawnedServingValidationResult {
  self.serving
}

///|
pub fn RuntimeSpawnedBenchmarkResult::startup_measurement(
  self : RuntimeSpawnedBenchmarkResult,
) -> RuntimeStartupMeasurement {
  self.startup
}
