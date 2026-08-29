///|
fn fused_benchmark_expected_order(
  trial_ordinal : Int,
) -> FusedCandidateBenchmarkPairOrder {
  if trial_ordinal % 2 == 0 {
    BaselineThenCandidate
  } else {
    CandidateThenBaseline
  }
}

///|
fn fused_benchmark_validate_protocol(
  identity : FusedBenchmarkCandidateIdentity,
  protocol : FusedCandidateBenchmarkProtocol,
) -> Unit raise FusedCandidateBenchmarkError {
  if protocol.candidate != identity.candidate {
    raise MismatchedFusedCandidateBenchmark(CandidateIdentity)
  }
  if protocol.shape_set != fused_candidate_benchmark_shape_set_identity() {
    raise MismatchedFusedCandidateBenchmark(ShapeCoverage)
  }
  if protocol.warmup_iterations != FUSED_BENCHMARK_WARMUP_ITERATIONS ||
    protocol.trials_per_shape != FUSED_BENCHMARK_TRIALS_PER_SHAPE ||
    protocol.repetitions_per_arm != FUSED_BENCHMARK_REPETITIONS_PER_ARM {
    raise MismatchedFusedCandidateBenchmark(Protocol)
  }
  if identity.target_compute_major != 12 ||
    identity.target_compute_minor != 0 ||
    !identity.target_bf16 ||
    !identity.target_cublas_lt ||
    identity.max_query_rows != 1 ||
    identity.max_query_tokens != 8 ||
    identity.max_page_table_entries != 4 ||
    identity.grid_x != 8 ||
    identity.grid_y != 1 ||
    identity.grid_z != 1 ||
    identity.block_x != 128 ||
    identity.block_y != 1 ||
    identity.block_z != 1 ||
    identity.fused_multiply_add ||
    identity.reassociate {
    raise MismatchedFusedCandidateBenchmark(CandidateIdentity)
  }
}

///|
fn fused_benchmark_validate_arm_resources(
  observation : FusedCandidateBenchmarkArmObservation,
) -> Unit raise FusedCandidateBenchmarkError {
  if observation.resources_acquired != observation.resources_released ||
    observation.live_resources_before != observation.live_resources_after ||
    observation.live_bytes_before != observation.live_bytes_after ||
    observation.resource_failures != 0L {
    raise MismatchedFusedCandidateBenchmark(ResourceBalance)
  }
}

///|
fn fused_benchmark_validate_observation(
  identity : FusedBenchmarkCandidateIdentity,
  protocol : FusedCandidateBenchmarkProtocol,
  expected_shape : FusedCandidateBenchmarkShape,
  expected_trial : Int,
  observation : FusedCandidateBenchmarkPairedObservation,
) -> Unit raise FusedCandidateBenchmarkError {
  if observation.shape != expected_shape {
    raise MismatchedFusedCandidateBenchmark(ShapeCoverage)
  }
  if observation.trial_ordinal != expected_trial ||
    observation.order != fused_benchmark_expected_order(expected_trial) ||
    observation.repetitions != protocol.repetitions_per_arm {
    raise MismatchedFusedCandidateBenchmark(ObservationOrder)
  }
  let expected_work = Int64::from_int(
    observation.repetitions * fused_benchmark_shape_tokens(expected_shape),
  )
  if observation.baseline.work_units != expected_work ||
    observation.candidate.work_units != expected_work {
    raise MismatchedFusedCandidateBenchmark(WorkCounter)
  }
  fused_benchmark_validate_arm_resources(observation.baseline)
  fused_benchmark_validate_arm_resources(observation.candidate)
  if observation.baseline.dispatch_canary_before !=
    observation.baseline.dispatch_canary_after {
    raise MismatchedFusedCandidateBenchmark(DispatchCanary)
  }
  let expected_canary_delta = identity.dispatch_canary *
    observation.repetitions.reinterpret_as_uint() *
    fused_benchmark_shape_tokens(expected_shape).reinterpret_as_uint()
  if observation.candidate.dispatch_canary_after -
    observation.candidate.dispatch_canary_before !=
    expected_canary_delta {
    raise MismatchedFusedCandidateBenchmark(DispatchCanary)
  }
  if observation.candidate.duration_ns >= observation.baseline.duration_ns {
    raise MismatchedFusedCandidateBenchmark(WinPolicy)
  }
}

///|
fn fused_benchmark_validate_observations(
  identity : FusedBenchmarkCandidateIdentity,
  protocol : FusedCandidateBenchmarkProtocol,
  observations : ArrayView[FusedCandidateBenchmarkPairedObservation],
) -> Array[FusedBenchmarkShapeSummary] raise FusedCandidateBenchmarkError {
  let shapes = fused_benchmark_shapes()
  let expected_count = shapes.length() * protocol.trials_per_shape
  if observations.length() != expected_count {
    raise MismatchedFusedCandidateBenchmark(ShapeCoverage)
  }
  let summaries : Array[FusedBenchmarkShapeSummary] = []
  for shape_index, shape in shapes {
    let baseline_durations : Array[Int64] = []
    let candidate_durations : Array[Int64] = []
    let mut baseline_work_total = 0L
    let mut candidate_work_total = 0L
    for trial in 0..<protocol.trials_per_shape {
      let observation = observations[shape_index * protocol.trials_per_shape +
        trial]
      fused_benchmark_validate_observation(
        identity, protocol, shape, trial, observation,
      )
      baseline_durations.push(observation.baseline.duration_ns)
      candidate_durations.push(observation.candidate.duration_ns)
      baseline_work_total = fused_benchmark_checked_add(
        baseline_work_total,
        observation.baseline.work_units,
        WorkCounter,
      )
      candidate_work_total = fused_benchmark_checked_add(
        candidate_work_total,
        observation.candidate.work_units,
        WorkCounter,
      )
    }
    let baseline = fused_benchmark_summarize(baseline_durations)
    let candidate = fused_benchmark_summarize(candidate_durations)
    if candidate.total > 9223372036854775807L / 100L ||
      baseline.total >
      9223372036854775807L / (100L - FUSED_BENCHMARK_MINIMUM_WIN_PERCENT) {
      raise OverflowedFusedCandidateBenchmark(WinPolicy)
    }
    if candidate.total * 100L >
      baseline.total * (100L - FUSED_BENCHMARK_MINIMUM_WIN_PERCENT) {
      raise MismatchedFusedCandidateBenchmark(WinPolicy)
    }
    summaries.push({
      shape,
      baseline,
      candidate,
      baseline_work_total,
      candidate_work_total,
    })
  }
  summaries
}
