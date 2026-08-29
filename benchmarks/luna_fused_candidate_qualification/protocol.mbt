///|
const FUSED_BENCHMARK_WARMUP_ITERATIONS : Int = 16

///|
const FUSED_BENCHMARK_TRIALS_PER_SHAPE : Int = 8

///|
const FUSED_BENCHMARK_REPETITIONS_PER_ARM : Int = 64

///|
const FUSED_BENCHMARK_MINIMUM_WIN_PERCENT : Int64 = 5L

///|
pub struct FusedCandidateBenchmarkProtocol {
  priv candidate : @luna_cuda_fused_parallel_aot.FusedParallelDigest
  priv shape_set : @luna_cuda_fused_parallel_aot.FusedParallelDigest
  priv clock : @luna_cuda_fused_parallel_aot.FusedParallelDigest
  priv resource_collector : @luna_cuda_fused_parallel_aot.FusedParallelDigest
  priv warmup_iterations : Int
  priv trials_per_shape : Int
  priv repetitions_per_arm : Int
}

///|
/// Declares caller-supplied raw observations for one exact candidate. The
/// protocol carries identities only; it cannot execute or time a benchmark.
pub fn FusedCandidateBenchmarkProtocol::new(
  candidate~ : @luna_cuda_fused_parallel_aot.FusedParallelDigest,
  shape_set~ : @luna_cuda_fused_parallel_aot.FusedParallelDigest,
  clock~ : @luna_cuda_fused_parallel_aot.FusedParallelDigest,
  resource_collector~ : @luna_cuda_fused_parallel_aot.FusedParallelDigest,
  warmup_iterations~ : Int,
  trials_per_shape~ : Int,
  repetitions_per_arm~ : Int,
) -> FusedCandidateBenchmarkProtocol raise FusedCandidateBenchmarkError {
  if warmup_iterations <= 0 ||
    warmup_iterations > 1000000 ||
    trials_per_shape <= 0 ||
    trials_per_shape > 1024 ||
    repetitions_per_arm <= 0 ||
    repetitions_per_arm > 1000000 ||
    candidate == shape_set ||
    candidate == clock ||
    candidate == resource_collector ||
    shape_set == clock ||
    shape_set == resource_collector ||
    clock == resource_collector {
    raise InvalidFusedCandidateBenchmark(Protocol)
  }
  {
    candidate,
    shape_set,
    clock,
    resource_collector,
    warmup_iterations,
    trials_per_shape,
    repetitions_per_arm,
  }
}

///|
pub fn fused_candidate_benchmark_warmup_iterations() -> Int {
  FUSED_BENCHMARK_WARMUP_ITERATIONS
}

///|
pub fn fused_candidate_benchmark_trials_per_shape() -> Int {
  FUSED_BENCHMARK_TRIALS_PER_SHAPE
}

///|
pub fn fused_candidate_benchmark_repetitions_per_arm() -> Int {
  FUSED_BENCHMARK_REPETITIONS_PER_ARM
}

///|
pub fn fused_candidate_benchmark_minimum_win_percent() -> Int64 {
  FUSED_BENCHMARK_MINIMUM_WIN_PERCENT
}
